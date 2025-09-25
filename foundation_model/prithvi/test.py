#!/usr/bin/env python3
# eval_prithvi.py
import os, sys, csv, json, argparse, logging
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchmetrics.detection import MeanAveragePrecision


# -------------------------
# Logging
# -------------------------
def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "evaluate.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


# -------------------------
# Dataset (PNG RGB + YOLO-OBB->AABB)
# -------------------------
class BrickKilnDataset(Dataset):
    def __init__(self, root: str, split: str, input_size: int = 224):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images"
        self.label_dir = self.root / "yolo_obb_labels"
        self.input_size = input_size

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])

        self.img_files = []
        all_pngs = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(".png")])
        logging.info(f"Scanning {len(all_pngs)} PNGs in {self.img_dir}...")
        for fname in tqdm(all_pngs, desc=f"Verify {split} data"):
            label_path = self.label_dir / f"{Path(fname).stem}.txt"
            if label_path.exists():
                with open(label_path, "r") as f:
                    for line in f:
                        if len(line.strip().split()) == 9:
                            self.img_files.append(fname)
                            break
        logging.info(f"Found {len(self.img_files)} valid images.")

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx: int):
        img_name = self.img_files[idx]
        img = Image.open(self.img_dir / img_name).convert("RGB")
        img_t = self.transform(img)
        _, h, w = img_t.shape

        boxes, labels = [], []
        with open(self.label_dir / f"{Path(img_name).stem}.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                cls_id = int(parts[0]) + 1
                obb = np.array([float(p) for p in parts[1:]], dtype=np.float32)
                xs, ys = obb[0::2] * w, obb[1::2] * h
                xmin, ymin = float(np.min(xs)), float(np.min(ys))
                xmax, ymax = float(np.max(xs)), float(np.max(ys))
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cls_id)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "filename": img_name,
        }
        return img_name, img_t, target


def collate_fn_with_name(batch):
    batch = [b for b in batch if b[2]["boxes"].shape[0] > 0]
    if not batch:
        return None, None, None
    names = [b[0] for b in batch]
    imgs  = [b[1] for b in batch]
    tgts  = [b[2] for b in batch]
    return names, imgs, tgts


# -------------------------
# Robust local Prithvi import
# -------------------------
def import_prithvi_mae(prithvi_dir: str):
    prithvi_dir = os.path.abspath(prithvi_dir)
    if prithvi_dir not in sys.path:
        sys.path.insert(0, prithvi_dir)
    try:
        from prithvi_mae import PrithviMAE  # packaged layout
        return PrithviMAE
    except Exception:
        import importlib.util
        candidates = [
            os.path.join(prithvi_dir, "prithvi_mae.py"),
            os.path.join(prithvi_dir, "src", "prithvi_mae.py"),
            os.path.join(prithvi_dir, "prithvi_100m", "prithvi_mae.py"),
        ]
        for c in candidates:
            if os.path.exists(c):
                spec = importlib.util.spec_from_file_location("prithvi_mae", c)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod.PrithviMAE
        raise RuntimeError(f"Could not import prithvi_mae from {prithvi_dir}")


# -------------------------
# Prithvi backbone wrapper (matches your training)
# -------------------------
class PrithviBackboneWrapper(nn.Module):
    def __init__(self, prithvi_dir: str, ckpt_name: str, image_size: int, freeze_backbone: bool = False):
        super().__init__()
        self.image_size = image_size

        PrithviMAE = import_prithvi_mae(prithvi_dir)

        # read config.yaml
        import yaml
        cfg_path = os.path.join(prithvi_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config.yaml in {prithvi_dir}")
        with open(cfg_path, "r") as f:
            y = yaml.safe_load(f)
        margs = y.get("model_args", {})
        margs.update({
            "img_size": image_size,
            "num_frames": 1,
            "in_chans": 6,
            "encoder_only": True,
        })

        self.prithvi = PrithviMAE(**margs)

        # load checkpoint
        ckpt_path = os.path.join(prithvi_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        for k in list(state_dict.keys()):
            if "pos_embed" in k:
                del state_dict[k]
        missing, unexpected = self.prithvi.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded Prithvi weights. Missing={len(missing)}, Unexpected={len(unexpected)}")

        self.hidden_dim = margs.get("embed_dim", 768)
        self.patch_size = margs.get("patch_size", 16)

        self.adapter = nn.Conv2d(3, 6, kernel_size=1, bias=True)

        if freeze_backbone:
            for p in self.prithvi.parameters():
                p.requires_grad = False
            for p in self.adapter.parameters():
                p.requires_grad = True

        self.out_channels = self.hidden_dim
        self.down4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 2, 1)
        self.down5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 2, 1)
        self.down6 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 2, 1)

        self.norm3 = nn.GroupNorm(32, self.hidden_dim)
        self.norm4 = nn.GroupNorm(32, self.hidden_dim)
        self.norm5 = nn.GroupNorm(32, self.hidden_dim)
        self.norm6 = nn.GroupNorm(32, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        B, C, H, W = x.shape
        assert H == W == self.image_size
        assert H % self.patch_size == 0
        x6 = self.adapter(x)
        x5d = x6.unsqueeze(2)  # add time dim (T=1)

        with torch.cuda.amp.autocast(enabled=torch.is_autocast_enabled()):
            feats_seq = self.prithvi.encoder.forward_features(x5d)  # list of [B, 1+L, D]
            maps = self.prithvi.encoder.prepare_features_for_image_model(feats_seq)  # list of [B, D*T, h, w]
            feat = maps[-1]  # [B, D, H', W'] (since T=1)

            p3 = self.norm3(feat)
            p4 = self.norm4(self.down4(p3))
            p5 = self.norm5(self.down5(p4))
            p6 = self.norm6(self.down6(p5))

        return OrderedDict({"0": p3, "1": p4, "2": p5, "3": p6})


def create_model(image_size: int, prithvi_dir: str, ckpt_name: str, num_classes: int = 4):
    backbone = PrithviBackboneWrapper(prithvi_dir, ckpt_name, image_size=image_size, freeze_backbone=False)
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )
    roi_pooler = MultiScaleRoIAlign(["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    # match your training normalization for Prithvi (you used ImageNet stats in the latest script)
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=image_size,
        max_size=image_size,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    return model


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, data_loader, device, output_dir, score_thr=0.0):
    model.eval()

    metric_global = MeanAveragePrecision(box_format='xyxy', class_metrics=False)                  # 0.5:0.95
    metric_classwise_50 = MeanAveragePrecision(box_format='xyxy', class_metrics=True, iou_thresholds=[0.5])
    metric_agnostic_50 = MeanAveragePrecision(box_format='xyxy', class_metrics=False, iou_thresholds=[0.5])

    class_names = {1: 'CFCBK', 2: 'FCBK', 3: 'Zigzag'}
    rows = []

    for names, images, targets in tqdm(data_loader, desc="Evaluating (Prithvi)"):
        if images is None:
            continue
        images = [im.to(device) for im in images]
        preds = model(images)

        preds_cpu = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        tgts_cpu  = [{k: (v if k == "filename" else v.detach().cpu()) for k, v in t.items()} for t in targets]

        pure_targets = [{k: v for k, v in t.items() if k in ("boxes","labels")} for t in tgts_cpu]
        metric_global.update(preds_cpu, pure_targets)
        metric_classwise_50.update(preds_cpu, pure_targets)

        agn_preds = [{'boxes': p['boxes'], 'scores': p['scores'], 'labels': torch.ones_like(p['labels'])} for p in preds_cpu]
        agn_targs = [{'boxes': t['boxes'], 'labels': torch.ones_like(t['labels'])} for t in pure_targets]
        metric_agnostic_50.update(agn_preds, agn_targs)

        for name, t, p in zip(names, pure_targets, preds_cpu):
            # GT rows
            for b, lb in zip(t["boxes"].tolist(), t["labels"].tolist()):
                rows.append([name, "ground_truth", int(lb), class_names.get(int(lb), "N/A"), 1.0,
                             float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            # PRED rows
            for b, lb, s in zip(p["boxes"].tolist(), p["labels"].tolist(), p["scores"].tolist()):
                if s >= score_thr:
                    rows.append([name, "prediction", int(lb), class_names.get(int(lb), "N/A"), float(s),
                                 float(b[0]), float(b[1]), float(b[2]), float(b[3])])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "test_results_final.csv"
    pd.DataFrame(rows, columns=["image_name","box_type","class_id","class_name","confidence_score",
                                "bbox_xmin","bbox_ymin","bbox_xmax","bbox_ymax"]).to_csv(csv_path, index=False)
    logging.info(f"Saved detections CSV -> {csv_path}")

    # Metrics
    res_g  = metric_global.compute()
    res_cw = metric_classwise_50.compute()
    res_ag = metric_agnostic_50.compute()

    map_all = float(res_g.get('map', torch.tensor(0.)).item())
    map50_g = float(res_g.get('map_50', torch.tensor(0.)).item())
    ca50    = float(res_ag.get('map_50', torch.tensor(0.)).item())

    class_ids = res_cw.get('classes', torch.tensor([])).tolist()
    map_pc   = res_cw.get('map_per_class', torch.tensor([])).tolist()
    class_map50 = {int(cid): float(m) for cid, m in zip(class_ids, map_pc)}

    logging.info("\n" + "="*64)
    logging.info("    SENTINELKILNDB — PRITHVI (Test) Results")
    logging.info("="*64)
    logging.info(f"mAP(0.5:0.95): {100*map_all:6.2f}")
    logging.info(f"mAP@50 (global): {100*map50_g:6.2f}")
    logging.info(f"CA mAP@50 (class-agnostic): {100*ca50:6.2f}")
    logging.info("-"*64)
    logging.info(f"CFCBK mAP@50: {100*class_map50.get(1,0.0):6.2f} | "
                 f"FCBK mAP@50: {100*class_map50.get(2,0.0):6.2f} | "
                 f"Zigzag mAP@50: {100*class_map50.get(3,0.0):6.2f}")
    logging.info("="*64 + "\n")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    ds = BrickKilnDataset(args.test_path, "test", args.input_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_fn_with_name, pin_memory=True)

    model = create_model(image_size=args.input_size,
                         prithvi_dir=args.prithvi_dir,
                         ckpt_name=args.ckpt_name,
                         num_classes=4).to(device)

    logging.info(f"Loading weights: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    logging.info(f"Loaded. Missing={len(missing)}, Unexpected={len(unexpected)}")

    evaluate(model, dl, device, args.output_dir, score_thr=args.score_thr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_path",   type=str, required=True)
    p.add_argument("--weights",     type=str, required=True, help="best_model.pth from Prithvi training")
    p.add_argument("--output_dir",  type=str, required=True)
    p.add_argument("--device",      type=str, default="cuda:0")
    p.add_argument("--input_size",  type=int, default=224)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--score_thr",   type=float, default=0.0)
    p.add_argument("--prithvi_dir", type=str, required=True, help="folder containing prithvi_mae.py & config.yaml")
    p.add_argument("--ckpt_name",   type=str, default="Prithvi_100M.pt")
    args = p.parse_args()
    main(args)
