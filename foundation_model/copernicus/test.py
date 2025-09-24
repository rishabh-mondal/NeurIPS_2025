#!/usr/bin/env python3
# eval_copernicus_classwise_csv.py
import os, sys, csv, json, logging, argparse, numpy as np
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchmetrics.detection import MeanAveragePrecision

# -------------------------
# TorchGeo CopernicusFM
# -------------------------
try:
    from torchgeo.models import CopernicusFM, copernicusfm_base, CopernicusFM_Base_Weights
except ImportError as e:
    print(f"❌ TorchGeo not found or too old for CopernicusFM: {e}\nTry: pip install 'torchgeo>=0.7'")
    sys.exit(1)

# -------------------------
# Logging
# -------------------------
def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'copernicus_evaluate2.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )

# -------------------------
# Dataset + OBB->AABB
# -------------------------
class BrickKilnDataset(Dataset):
    def __init__(self, root: str, split: str, input_size: int = 224):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images"
        self.label_dir = self.root / "yolo_obb_labels"

        # match your Copernicus training normalization
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        self.img_files = []
        all_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        logging.info(f"Scanning {len(all_files)} images in {self.img_dir}...")
        for img_name in tqdm(all_files, desc=f"Verify {split} data"):
            if self._has_valid_annotations(img_name):
                self.img_files.append(img_name)
        logging.info(f"Found {len(self.img_files)} valid images in {self.img_dir}")

    def _has_valid_annotations(self, img_name: str) -> bool:
        label_path = self.label_dir / f"{Path(img_name).stem}.txt"
        if not label_path.exists(): return False
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
                if len(parts) != 9:  # class + 8 coords (normalized)
                    continue
                class_id = int(parts[0]) + 1
                obb = np.array([float(p) for p in parts[1:]])
                xs, ys = obb[0::2] * w, obb[1::2] * h
                xmin, ymin, xmax, ymax = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "filename": img_name,
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
# CopernicusFM Backbone wrapper (matches your training)
# -------------------------
class CopernicusFMBackboneWrapper(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = copernicusfm_base(weights=CopernicusFM_Base_Weights.CopernicusFM_ViT) if pretrained else CopernicusFM()
        for p in self.model.parameters():
            p.requires_grad = False

        self.wavelengths = [640, 550, 470]
        self.bandwidths  = [100, 90, 80]

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            if hasattr(self.model, 'patch_embed_spectral'):
                wvs = torch.tensor(self.wavelengths, dtype=torch.float32)
                bws = torch.tensor(self.bandwidths, dtype=torch.float32)
                patch_emb = self.model.patch_embed_spectral(dummy, wavelengths=wvs, bandwidths=bws)  # [B,N,D]
                D = patch_emb.shape[-1]
            else:
                meta = torch.full((1,4), float('nan'))
                feats = self.model.forward_features(
                    dummy, meta, wavelengths=self.wavelengths, bandwidths=self.bandwidths, input_mode='spectral'
                )
                D = feats.shape[-1] if feats.ndim == 2 else feats.shape[-1]

        self.feature_adapter = nn.Sequential(
            nn.Conv2d(D, 256, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 768, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.adapter_out_channels = 768
        self.out_channels = 768

    def forward(self, x):
        B = x.shape[0]
        if hasattr(self.model, 'patch_embed_spectral'):
            wvs = torch.tensor(self.wavelengths, device=x.device).float()
            bws = torch.tensor(self.bandwidths, device=x.device).float()
            tokens = self.model.patch_embed_spectral(x, wavelengths=wvs, bandwidths=bws)  # [B,N,D]
            grid = int(np.sqrt(tokens.shape[1]))
            spatial = tokens.transpose(1, 2).reshape(B, tokens.shape[-1], grid, grid)
        else:
            meta = torch.full((B,4), float('nan'), device=x.device)
            feats = self.model.forward_features(
                x, meta, wavelengths=self.wavelengths, bandwidths=self.bandwidths, input_mode='spectral'
            )
            if feats.ndim == 2:
                grid = 14
                spatial = feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, grid, grid)
            else:
                grid = int(np.sqrt(feats.shape[1]))
                spatial = feats.transpose(1, 2).reshape(B, feats.shape[-1], grid, grid)

        adapted = self.feature_adapter(spatial)
        out = OrderedDict()
        cur = adapted
        for i in range(4):
            out[str(i)] = cur
            if i < 3:
                h2 = max(1, cur.shape[2] // 2)
                w2 = max(1, cur.shape[3] // 2)
                cur = nn.functional.interpolate(cur, size=(h2,w2), mode='bilinear', align_corners=False)
        return out

def create_model(num_classes: int = 4, pretrained: bool = True):
    backbone = CopernicusFMBackboneWrapper(pretrained=pretrained)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    roi_pooler = MultiScaleRoIAlign(['0','1','2','3'], output_size=7, sampling_ratio=2)
    return FasterRCNN(
        backbone, num_classes=num_classes,
        rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler
    )

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, data_loader, device, output_dir, score_thr=0.0, save_json=None):
    model.eval()

    metric_global = MeanAveragePrecision(box_format='xyxy', class_metrics=False)
    metric_classwise_50 = MeanAveragePrecision(box_format='xyxy', class_metrics=True, iou_thresholds=[0.5])
    metric_agnostic_50  = MeanAveragePrecision(box_format='xyxy', class_metrics=False, iou_thresholds=[0.5])

    class_names = {1:'CFCBK', 2:'FCBK', 3:'Zigzag'}
    rows, coco_results = [], []

    for batch in tqdm(data_loader, desc="Evaluating (Copernicus)"):
        img_names, images, targets = batch
        if images is None: continue

        images = [img.to(device) for img in images]
        preds = model(images)

        preds_cpu = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        t_cpu = [{k: (v if k == "filename" else v.detach().cpu()) for k, v in t.items()} for t in targets]

        pure_targets = [{k: v for k, v in t.items() if k in ("boxes","labels")} for t in t_cpu]
        metric_global.update(preds_cpu, pure_targets)
        metric_classwise_50.update(preds_cpu, pure_targets)

        agn_preds = [{'boxes': p['boxes'], 'scores': p['scores'], 'labels': torch.ones_like(p['labels'])} for p in preds_cpu]
        agn_targs = [{'boxes': t['boxes'], 'labels': torch.ones_like(t['labels'])} for t in pure_targets]
        metric_agnostic_50.update(agn_preds, agn_targs)

        for name, t, p in zip(img_names, pure_targets, preds_cpu):
            for b, lb in zip(t["boxes"].tolist(), t["labels"].tolist()):
                rows.append([name, "ground_truth", int(lb), class_names.get(int(lb), "N/A"), 1.0,
                             float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            for b, lb, s in zip(p["boxes"].tolist(), p["labels"].tolist(), p["scores"].tolist()):
                if s >= score_thr:
                    rows.append([name, "prediction", int(lb), class_names.get(int(lb), "N/A"), float(s),
                                 float(b[0]), float(b[1]), float(b[2]), float(b[3])])

            if save_json is not None:
                img_id = int(t["image_id"].item())
                for b, s, lb in zip(p["boxes"].tolist(), p["scores"].tolist(), p["labels"].tolist()):
                    x1, y1, x2, y2 = b
                    coco_results.append({
                        "image_id": img_id,
                        "category_id": int(lb),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s)
                    })

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

    res_g  = metric_global.compute()
    res_cw = metric_classwise_50.compute()
    res_ag = metric_agnostic_50.compute()

    map_all = float(res_g.get('map', torch.tensor(0.0)).item())
    map50   = float(res_g.get('map_50', torch.tensor(0.0)).item())
    ca50    = float(res_ag.get('map_50', torch.tensor(0.0)).item())

    cls_ids = res_cw.get('classes', torch.tensor([])).tolist()
    map_pc  = res_cw.get('map_per_class', torch.tensor([])).tolist()
    class_map50 = {int(cid): float(m) for cid, m in zip(cls_ids, map_pc)}

    pct = lambda x: 100.0 * x
    logging.info("\n" + "="*66)
    logging.info("    SENTINELKILNDB — CopernicusFM (Test) Results")
    logging.info("="*66)
    logging.info(f"mAP(0.5:0.95): {pct(map_all):6.2f}")
    logging.info(f"mAP@50 (global): {pct(map50):6.2f}")
    logging.info(f"CA mAP@50 (class-agnostic): {pct(ca50):6.2f}")
    logging.info("-"*66)
    logging.info(f"CFCBK mAP@50: {pct(class_map50.get(1,0.0)):6.2f} | "
                 f"FCBK mAP@50: {pct(class_map50.get(2,0.0)):6.2f} | "
                 f"Zigzag mAP@50: {pct(class_map50.get(3,0.0)):6.2f}")
    logging.info("="*66 + "\n")

    return map_all, map50, ca50, class_map50

# -------------------------
# Main
# -------------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    test_dataset = BrickKilnDataset(args.test_path, 'test', args.input_size)
    test_loader  = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn_with_name, pin_memory=True
    )

    model = create_model(num_classes=4, pretrained=True).to(device)

    logging.info(f"Loading weights from: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    logging.info(f"Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    json_out = os.path.join(args.output_dir, "copernicus_coco_results.json") if args.save_json else None
    map_all, map50, ca50, class_map50 = evaluate(
        model, test_loader, device, output_dir=args.output_dir,
        score_thr=args.score_thr, save_json=json_out
    )

    print(f"[TEST][Copernicus] mAP(0.5:0.95): {100*map_all:.2f} | mAP@50: {100*map50:.2f} | "
          f"CA mAP@50: {100*ca50:.2f} | "
          f"CFCBK: {100*class_map50.get(1,0):.2f}, FCBK: {100*class_map50.get(2,0):.2f}, Zigzag: {100*class_map50.get(3,0):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path',  type=str, default="/sentinelkilndb_bechmarking_data/test")
    parser.add_argument('--weights',    type=str, default="/work_dirs/copernicus_train/best_copernicus_model.pth")
    parser.add_argument('--output_dir', type=str, default="/work_dirs/copernicus_eval_csv")
    parser.add_argument('--device',     type=str, default="cuda:0")

    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size',   type=int, default=16)
    parser.add_argument('--num_workers',  type=int, default=8)
    parser.add_argument('--score_thr',    type=float, default=0.0, help="Score threshold for *logging only*")
    parser.add_argument('--save_json',    action='store_true')

    args = parser.parse_args()
    main(args)
