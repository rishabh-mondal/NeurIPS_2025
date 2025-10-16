#!/usr/bin/env python3
# eval_croma_classwise_csv.py
import os, sys, csv, math, json, itertools, logging, argparse
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics.detection import MeanAveragePrecision
import torch.nn as nn

# -------------------------
# TorchGeo CROMA
# -------------------------
try:
    from torchgeo.models.croma import croma_base, CROMABase_Weights
except ImportError:
    print("Error: torchgeo is not installed or too old for CROMA. Try: pip install 'torchgeo>=0.7'")
    sys.exit(1)

# -------------------------
# Logging
# -------------------------
def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'evaluate2.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )

# -------------------------
# Dataset (PNG RGB only) + YOLO-OBB (9-tuple) -> AABB
# -------------------------
class BrickKilnDataset(Dataset):
    def __init__(self, root: str, split: str, input_size: int = 120):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images"
        self.label_dir = self.root / "yolo_obb_labels"

        # no normalization for CROMA (to match your training)
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])

        self.img_files = []
        all_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(".png")])
        logging.info(f"Scanning {len(all_files)} PNGs in {self.img_dir}...")
        for img_name in tqdm(all_files, desc=f"Verify {split} data"):
            if self._has_valid_annotations(img_name):
                self.img_files.append(img_name)
        logging.info(f"Found {len(self.img_files)} valid images in {self.img_dir}")

    def _has_valid_annotations(self, img_name: str) -> bool:
        label_path = self.label_dir / f"{Path(img_name).stem}.txt"
        if not label_path.exists():
            return False
        with open(label_path, 'r') as f:
            for line in f:
                if len(line.strip().split()) == 9:
                    return True
        return False

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx: int):
        img_name = self.img_files[idx]
        img_path = self.img_dir / img_name
        label_path = self.label_dir / f"{Path(img_name).stem}.txt"

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        _, h, w = img_tensor.shape

        boxes, labels = [], []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                cls_id = int(parts[0]) + 1  # reserve 0 for background
                obb = np.array([float(p) for p in parts[1:]])
                xs, ys = obb[0::2] * w, obb[1::2] * h
                xmin, ymin, xmax, ymax = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cls_id)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "filename": img_name
        }
        return img_name, img_tensor, target

def collate_fn_with_name(batch):
    batch = [item for item in batch if item[2]["boxes"].shape[0] > 0]
    if not batch: return None, None, None
    img_names = [b[0] for b in batch]
    images = [b[1] for b in batch]
    targets = [b[2] for b in batch]
    return img_names, images, targets

# -------------------------
# 2D ALiBi
# -------------------------
def _get_2dalibi(num_heads: int, num_patches: int) -> torch.Tensor:
    side = int(math.sqrt(num_patches))
    pts = list(itertools.product(range(side), range(side)))

    def _slopes(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    slopes = torch.tensor(_slopes(num_heads)).unsqueeze(1)
    idxs = []
    for p1 in pts:
        for p2 in pts:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, num_heads, num_patches, num_patches)

# -------------------------
# CROMA Backbone Wrapper
# -------------------------
class CromaBackboneWrapper(nn.Module):
    def __init__(self, image_size: int = 120, freeze_croma: bool = False):
        super().__init__()
        assert image_size % 8 == 0, "CROMA expects image_size to be a multiple of 8"
        self.image_size = image_size
        self.rgb_to_ms = nn.Conv2d(3, 12, kernel_size=1, bias=True)
        self.croma = croma_base(
            weights=CROMABase_Weights.CROMA_VIT,
            modalities=['optical'],
            image_size=image_size,
        )
        if freeze_croma:
            for p in self.croma.parameters():
                p.requires_grad = False

        self.encoder_dim = self.croma.encoder_dim
        self.patch_size = self.croma.patch_size
        self.num_heads = self.croma.num_heads
        self.out_channels = self.encoder_dim

        self.down4 = nn.Conv2d(self.encoder_dim, self.encoder_dim, 3, 2, 1)
        self.down5 = nn.Conv2d(self.encoder_dim, self.encoder_dim, 3, 2, 1)
        self.down6 = nn.Conv2d(self.encoder_dim, self.encoder_dim, 3, 2, 1)

        self.norm3 = nn.GroupNorm(32, self.encoder_dim)
        self.norm4 = nn.GroupNorm(32, self.encoder_dim)
        self.norm5 = nn.GroupNorm(32, self.encoder_dim)
        self.norm6 = nn.GroupNorm(32, self.encoder_dim)

    def _ensure_attn_bias(self, H: int, device: torch.device):
        h_tokens = H // self.patch_size
        num_patches = h_tokens * h_tokens
        current = getattr(self.croma, "attn_bias", None)
        if (current is None) or (current.shape[-1] != num_patches):
            new_bias = _get_2dalibi(self.num_heads, num_patches).to(device)
            self.croma.attn_bias = new_bias

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        B, C, H, W = x.shape
        assert H == W, "CROMA expects square inputs"
        self._ensure_attn_bias(H, x.device)
        x12 = self.rgb_to_ms(x)
        out = self.croma(x_optical=x12)
        tokens = out["optical_encodings"]
        B, N, Cdim = tokens.shape
        side = int(math.isqrt(N))
        feat = tokens.permute(0, 2, 1).reshape(B, Cdim, side, side)

        p3 = self.norm3(feat)
        p4 = self.norm4(self.down4(p3))
        p5 = self.norm5(self.down5(p4))
        p6 = self.norm6(self.down6(p5))
        return OrderedDict({"0": p3, "1": p4, "2": p5, "3": p6})

# -------------------------
# Faster R-CNN
# -------------------------
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

def create_model(num_classes: int, image_size: int):
    backbone = CromaBackboneWrapper(image_size=image_size, freeze_croma=False)
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    roi_pooler = MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=image_size,
        max_size=image_size,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
    )

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, data_loader, device, output_dir, score_thr=0.0, save_json=None):
    model.eval()

    metric_global = MeanAveragePrecision(box_format='xyxy', class_metrics=False)                  # 0.5:0.95 + map_50
    metric_classwise_50 = MeanAveragePrecision(box_format='xyxy', class_metrics=True, iou_thresholds=[0.5])
    metric_agnostic_50 = MeanAveragePrecision(box_format='xyxy', class_metrics=False, iou_thresholds=[0.5])

    class_names = {1: 'CFCBK', 2: 'FCBK', 3: 'Zigzag'}
    rows = []
    coco_results = []

    for batch in tqdm(data_loader, desc="Evaluating (CROMA)"):
        img_names, images, targets = batch
        if images is None:
            continue

        images = [img.to(device) for img in images]
        preds = model(images)

        preds_cpu = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        t_cpu = [{k: (v if k == "filename" else v.detach().cpu()) for k, v in t.items()} for t in targets]

        # Update metrics (original labels)
        pure_targets = [{k: v for k, v in t.items() if k in ("boxes","labels")} for t in t_cpu]
        metric_global.update(preds_cpu, pure_targets)
        metric_classwise_50.update(preds_cpu, pure_targets)

        # Update metrics (class-agnostic view)
        agn_preds = [{'boxes': p['boxes'], 'scores': p['scores'], 'labels': torch.ones_like(p['labels'])} for p in preds_cpu]
        agn_targs = [{'boxes': t['boxes'], 'labels': torch.ones_like(t['labels'])} for t in pure_targets]
        metric_agnostic_50.update(agn_preds, agn_targs)

        # CSV rows: ground-truth then predictions (mirror Galileo)
        for name, t, p in zip(img_names, pure_targets, preds_cpu):
            # GT rows
            for b, lb in zip(t["boxes"].tolist(), t["labels"].tolist()):
                rows.append([name, "ground_truth", int(lb), class_names.get(int(lb), "N/A"), 1.0,
                             float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            # prediction rows (optionally threshold for logging only)
            for b, lb, s in zip(p["boxes"].tolist(), p["labels"].tolist(), p["scores"].tolist()):
                if s >= score_thr:
                    rows.append([name, "prediction", int(lb), class_names.get(int(lb), "N/A"), float(s),
                                 float(b[0]), float(b[1]), float(b[2]), float(b[3])])

            if save_json is not None:
                img_id = int(t_cpu[0]["image_id"].item())  # or t["image_id"] but not strictly needed per-file
                for b, s, lb in zip(p["boxes"].tolist(), p["scores"].tolist(), p["labels"].tolist()):
                    x1, y1, x2, y2 = b
                    coco_results.append({
                        "image_id": img_id,
                        "category_id": int(lb),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s)
                    })

    # Write CSV exactly like Galileo
    csv_path = Path(output_dir) / "test_results_final2.csv"
    os.makedirs(Path(output_dir), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name","box_type","class_id","class_name","confidence_score","bbox_xmin","bbox_ymin","bbox_xmax","bbox_ymax"])
        writer.writerows(rows)
    logging.info(f"Saved CSV to {csv_path}")

    if save_json is not None:
        with open(save_json, "w") as f:
            json.dump(coco_results, f)
        logging.info(f"Saved COCO JSON to {save_json}")

    # compute + pretty print
    res_g = metric_global.compute()
    res_cw = metric_classwise_50.compute()
    res_ag = metric_agnostic_50.compute()

    map_all = float(res_g.get('map', torch.tensor(0.)).item())        # 0.5:0.95
    map50_g = float(res_g.get('map_50', torch.tensor(0.)).item())     # sanity/global @0.5
    ca50    = float(res_ag.get('map_50', torch.tensor(0.)).item())    # agnostic @0.5

    class_ids = res_cw.get('classes', torch.tensor([])).tolist()
    map_pc   = res_cw.get('map_per_class', torch.tensor([])).tolist()
    class_map50 = {int(cid): float(m) for cid, m in zip(class_ids, map_pc)}

    pct = lambda x: 100.0 * x
    logging.info("\n" + "="*64)
    logging.info("    SENTINELKILNDB — CROMA (Test) Results")
    logging.info("="*64)
    logging.info(f"mAP(0.5:0.95): {pct(map_all):6.2f}")
    logging.info(f"mAP@50 (global): {pct(map50_g):6.2f}")
    logging.info(f"CA mAP@50 (class-agnostic): {pct(ca50):6.2f}")
    logging.info("-"*64)
    logging.info(f"CFCBK mAP@50: {pct(class_map50.get(1,0.0)):6.2f} | "
                 f"FCBK mAP@50: {pct(class_map50.get(2,0.0)):6.2f} | "
                 f"Zigzag mAP@50: {pct(class_map50.get(3,0.0)):6.2f}")
    logging.info("="*64 + "\n")

    return map_all, map50_g, ca50, class_map50

# -------------------------
# Main
# -------------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Data
    test_dataset = BrickKilnDataset(args.test_path, 'test', args.input_size)
    test_loader  = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn_with_name, pin_memory=True
    )

    # Model
    num_classes = 4
    model = create_model(num_classes=num_classes, image_size=args.input_size).to(device)

    # Load weights
    logging.info(f"Loading weights from: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    logging.info(f"Loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # Evaluate
    json_out = os.path.join(args.output_dir, "coco_results.json") if args.save_json else None
    map_all, map50, ca50, class_map50 = evaluate(
        model, test_loader, device, output_dir=args.output_dir,
        score_thr=args.score_thr, save_json=json_out
    )

    print(f"[TEST][CROMA] mAP(0.5:0.95): {100*map_all:.2f} | mAP@50: {100*map50:.2f} | "
          f"CA mAP@50: {100*ca50:.2f} | "
          f"CFCBK: {100*class_map50.get(1,0):.2f}, FCBK: {100*class_map50.get(2,0):.2f}, Zigzag: {100*class_map50.get(3,0):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path',  type=str, default="/sentinelkilndb_bechmarking_data/test")
    parser.add_argument('--weights',    type=str, default="/work_dirs/croma_train/best_model.pth")
    parser.add_argument('--output_dir', type=str, default="/work_dirs/croma_train")
    parser.add_argument('--device',     type=str, default="cuda:0")

    parser.add_argument('--input_size', type=int, default=120)  # multiple of 8
    parser.add_argument('--batch_size',   type=int, default=8)
    parser.add_argument('--num_workers',  type=int, default=8)
    parser.add_argument('--score_thr',    type=float, default=0.0, help="Score threshold for *logging only*")
    parser.add_argument('--save_json',    action='store_true')

    args = parser.parse_args()
    main(args)
