import os
import sys
import csv
import math
import json
import logging
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchmetrics.detection import MeanAveragePrecision

# -------------------------
# TorchGeo Panopticon
# -------------------------
try:
    from torchgeo.models.panopticon import (
        panopticon_vitb14,
        Panopticon_Weights,
    )
except ImportError:
    print("Error: torchgeo is not installed or too old for Panopticon. Try: pip install 'torchgeo>=0.7'")
    sys.exit(1)


# -------------------------
# Logging
# -------------------------
def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )


# -------------------------
# CSV Logger
# -------------------------
class CSVLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_map", "val_map50", "lr"])

    def log(self, epoch, train_loss, val_map, val_map50, lr):
        val_map = float('nan') if val_map is None else float(val_map)
        val_map50 = float('nan') if val_map50 is None else float(val_map50)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_map, val_map50, lr])


# -------------------------
# Dataset (PNG RGB only) + YOLO-OBB (9-tuple) -> AABB
# -------------------------
class BrickKilnDataset(Dataset):
    """
    root/
      images/*.png
      yolo_obb_labels/*.txt   (each line: cls x1 y1 x2 y2 x3 y3 x4 y4 in YOLO-OBB norm coords)
    Converts OBB to AABB for Faster R-CNN training/eval.
    """
    def __init__(self, root: str, split: str, input_size: int = 224):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images"
        self.label_dir = self.root / "yolo_obb_labels"

        assert input_size % 14 == 0, "Panopticon works best if input_size is a multiple of 14."
        self.input_size = input_size

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

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_name = self.img_files[idx]
        img_path = self.img_dir / img_name
        label_path = self.label_dir / f"{Path(img_name).stem}.txt"

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)  # [3,H,W]
        _, h, w = img_tensor.shape

        boxes, labels = [], []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                cls_id = int(parts[0]) + 1  # reserve 0 for background

                obb = np.array([float(p) for p in parts[1:]], dtype=np.float32)
                xs, ys = obb[0::2] * w, obb[1::2] * h

                # finite check
                if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
                    continue

                # clamp into image bounds
                xmin = float(np.clip(xs.min(), 0, w - 1))
                ymin = float(np.clip(ys.min(), 0, h - 1))
                xmax = float(np.clip(xs.max(), 0, w - 1))
                ymax = float(np.clip(ys.max(), 0, h - 1))

                # skip degenerate
                if xmax <= xmin or ymax <= ymin:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls_id)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_name": img_name,
        }
        return img_tensor, target


def collate_fn(batch):
    batch = [item for item in batch if item[1]["boxes"].shape[0] > 0]
    if not batch:
        return None, None
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


# -------------------------
# Panopticon Backbone Wrapper
# -------------------------
class PanopticonBackboneWrapper(nn.Module):
    """
    Wrap TorchGeo Panopticon (ViT-B/14) to provide feature maps for torchvision detectors.
    Returns a pyramid dict { '0','1','2','3' } for FPN/ROI head.
    """
    def __init__(self, image_size: int = 224, freeze_backbone: bool = False):
        super().__init__()
        assert image_size % 14 == 0, "Panopticon ViT-B/14 expects sizes multiple of 14."
        self.image_size = image_size

        # Load Panopticon with published weights at desired image size
        self.backbone = panopticon_vitb14(weights=Panopticon_Weights.VIT_BASE14, img_size=image_size)
        self.vit = self.backbone.model  # timm ViT with our Panopticon patch_embed
        self.encoder_dim = 768
        self.patch_size = 14

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Simple FPN-ish downsamples (keep channels constant)
        self.down4 = nn.Conv2d(self.encoder_dim, self.encoder_dim, kernel_size=3, stride=2, padding=1)
        self.down5 = nn.Conv2d(self.encoder_dim, self.encoder_dim, kernel_size=3, stride=2, padding=1)
        self.down6 = nn.Conv2d(self.encoder_dim, self.encoder_dim, kernel_size=3, stride=2, padding=1)

        self.norm3 = nn.GroupNorm(32, self.encoder_dim)
        self.norm4 = nn.GroupNorm(32, self.encoder_dim)
        self.norm5 = nn.GroupNorm(32, self.encoder_dim)
        self.norm6 = nn.GroupNorm(32, self.encoder_dim)

        self.out_channels = self.encoder_dim

        # Default RGB wavelengths (nm), order must match input channels
        self.register_buffer(
            "rgb_wavelengths",
            torch.tensor([665.0, 560.0, 492.0]).float(),
            persistent=False
        )

    def _build_x_dict(self, x: torch.Tensor) -> dict:
        B, C, H, W = x.shape
        assert C == 3, "This wrapper currently supports RGB inputs only."
        chn_ids = self.rgb_wavelengths.unsqueeze(0).expand(B, -1)  # [B,3]
        return {"imgs": x, "chn_ids": chn_ids}

    def _add_pos_embed(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """Add (and if needed interpolate) absolute pos embeddings to token sequence (w/o cls)."""
        vit = self.vit
        B, N, D = x_tokens.shape  # N = h*w
        pos = vit.pos_embed  # shape (1, 1 + N_ref, D)

        if pos is None:
            return x_tokens

        cls_pos = pos[:, :1]              # (1,1,D)
        tok_pos = pos[:, 1:]              # (1,N_ref,D)
        N_ref = tok_pos.shape[1]

        if N_ref != N:
            # interpolate token positional embeddings to new grid
            gs_ref = int(math.isqrt(N_ref))
            gs_new = int(math.isqrt(N))
            tok_pos_2d = tok_pos.reshape(1, gs_ref, gs_ref, D).permute(0, 3, 1, 2)  # (1,D,gs_ref,gs_ref)
            tok_pos_2d = F.interpolate(tok_pos_2d, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
            tok_pos = tok_pos_2d.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, D)  # (1,N,D)

        pos_new = torch.cat([cls_pos, tok_pos], dim=1)  # (1,1+N,D)
        return x_tokens + pos_new[:, 1:, :]

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        """
        Produce multiscale feature maps from per-patch tokens (pre-pooling).
        """
        B, C, H, W = x.shape
        assert H == W, "Panopticon wrapper expects square inputs"
        assert H % self.patch_size == 0, "Input size must be multiple of 14"

        vit = self.vit
        x_dict = self._build_x_dict(x)

        # 1) Patchify via PanopticonPE (handles channel embeddings/attention)
        tokens = vit.patch_embed(x_dict)  # (B, N, D), NO cls token yet

        # 2) Add abs positional embeddings (with safe interpolation)
        tokens = self._add_pos_embed(tokens)  # (B, N, D)

        # 3) Prepend cls token, dropout, transformer blocks + norm
        cls_tok = vit.cls_token.expand(B, -1, -1)              # (B,1,D)
        x_seq = torch.cat([cls_tok, tokens], dim=1)            # (B,1+N,D)
        x_seq = vit.pos_drop(x_seq)
        for blk in vit.blocks:
            x_seq = blk(x_seq)
        x_seq = vit.norm(x_seq)

        # 4) Remove cls, reshape tokens -> feature map
        tokens = x_seq[:, 1:, :]                               # (B, N, D)
        side = int(math.isqrt(tokens.shape[1]))                # H' = W' = H/14
        feat = tokens.permute(0, 2, 1).reshape(B, self.encoder_dim, side, side)

        # 5) Build simple pyramid
        p3 = self.norm3(feat)            # stride ~14
        p4 = self.norm4(self.down4(p3))  # ~28
        p5 = self.norm5(self.down5(p4))  # ~56
        p6 = self.norm6(self.down6(p5))  # ~112

        return OrderedDict({"0": p3, "1": p4, "2": p5, "3": p6})


# -------------------------
# Build Faster R-CNN
# -------------------------
def create_model(num_classes: int, image_size: int):
    backbone = PanopticonBackboneWrapper(image_size=image_size, freeze_backbone=False)

    anchor_generator = AnchorGenerator(
        sizes=( (32,), (64,), (128,), (256,),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # Keep fixed size (no 800 resize)
        min_size=image_size,
        max_size=image_size,
        # DINOv2/ViT stats
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )
    return model


# -------------------------
# Train / Validate / Test
# -------------------------
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    steps = 0
    for images, targets in tqdm(data_loader, desc="Training"):
        if images is None:
            continue
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        steps += 1

    return total_loss / max(1, steps)


@torch.no_grad()
def evaluate_loader(model, data_loader, device, class_metrics=False, save_raw=None):
    """
    Generic evaluator for val/test.
    If class_metrics=True -> per-class metrics too.
    If save_raw is a path, also save raw per-image detections to CSV.
    """
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=class_metrics)
    raw_rows = []
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        if images is None:
            continue
        images = [img.to(device) for img in images]
        preds = model(images)
        preds_cpu = [{k: v.to('cpu') for k, v in p.items()} for p in preds]
        t_cpu = [{k: (v.to('cpu') if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
        metric.update(preds_cpu, t_cpu)

        if save_raw is not None:
            for t, p in zip(t_cpu, preds_cpu):
                name = t.get("image_name", "unknown.png")
                boxes = p["boxes"].numpy() if len(p["boxes"]) else np.zeros((0,4))
                scores = p["scores"].numpy() if len(p["scores"]) else np.zeros((0,))
                labels = p["labels"].numpy() if len(p["labels"]) else np.zeros((0,))
                for b, s, l in zip(boxes, scores, labels):
                    raw_rows.append([name, float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                                     float(s), int(l)])

    res = metric.compute()
    if save_raw is not None:
        df = pd.DataFrame(raw_rows, columns=["image","x1","y1","x2","y2","score","label"])
        os.makedirs(os.path.dirname(save_raw), exist_ok=True)
        df.to_csv(save_raw, index=False)
    return res


def write_test_metrics(res_dict, out_overall_csv, out_perclass_csv, num_classes):
    overall = {
        "map": float(res_dict.get("map", torch.tensor(0.)).item()),
        "map_50": float(res_dict.get("map_50", torch.tensor(0.)).item()),
        "map_75": float(res_dict.get("map_75", torch.tensor(0.)).item()),
    }
    pd.DataFrame([overall]).to_csv(out_overall_csv, index=False)

    if "map_per_class" in res_dict and res_dict["map_per_class"] is not None:
        m = res_dict["map_per_class"].cpu().numpy()
        m50 = res_dict.get("map_50_per_class", torch.zeros(num_classes)).cpu().numpy()
        rows = []
        for cls in range(len(m)):
            rows.append({"class_id": int(cls), "map": float(m[cls]), "map_50": float(m50[cls])})
        pd.DataFrame(rows).to_csv(out_perclass_csv, index=False)


# -------------------------
# Checkpointing
# -------------------------
def save_checkpoint(path, model, optimizer, scheduler, epoch, best_map50):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_map50": best_map50,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"])
    if "cuda_rng_state_all" in ckpt and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
        except Exception:
            pass
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_map50 = float(ckpt.get("best_map50", 0.0))
    return start_epoch, best_map50


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    csv_logger = CSVLogger(os.path.join(args.output_dir, "results_2.csv"))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Data
    train_dataset = BrickKilnDataset(args.train_path, 'train', args.input_size)
    val_dataset   = BrickKilnDataset(args.val_path,   'val',   args.input_size)
    test_dataset  = BrickKilnDataset(args.test_path,  'test',  args.input_size) if args.test_path else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader  = None
    if test_dataset is not None:
        test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model
    num_classes = args.num_classes  # background + N classes (labels are 1..N)
    model = create_model(num_classes=num_classes, image_size=args.input_size).to(device)

    # Two LR groups: smaller LR for pretrained Panopticon; larger for heads
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone.backbone" in name:  # inside Panopticon/timm vit
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    logging.info(f"RCNN transform min_size={model.transform.min_size}, max_size={model.transform.max_size}")

    # Resume?
    start_epoch = 1
    best_map50 = 0.0
    if args.resume and os.path.isfile(args.resume):
        logging.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_map50 = load_checkpoint(args.resume, model, optimizer, lr_scheduler, device)
        logging.info(f"Resumed at epoch={start_epoch}, best_map50={best_map50:.4f}")

    # Train
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        res_val = evaluate_loader(model, val_loader, device, class_metrics=False)
        val_map = float(res_val.get('map', torch.tensor(0.)).item())
        val_map50 = float(res_val.get('map_50', torch.tensor(0.)).item())
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch} - Loss: {train_loss:.4f}, mAP: {val_map:.4f}, mAP@50: {val_map50:.4f} - lr: {current_lr:.6f}")
        csv_logger.log(epoch, train_loss, val_map, val_map50, current_lr)

        # Save rolling checkpoint
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch:03d}.pth")
        save_checkpoint(ckpt_path, model, optimizer, lr_scheduler, epoch, best_map50)

        # Optionally test every N epochs
        if test_loader is not None and (epoch % args.eval_test_every == 0 or epoch == args.epochs):
            ep_tag = f"epoch{epoch:03d}"
            out_overall = os.path.join(args.output_dir, f"test_overall_{ep_tag}.csv")
            out_percls  = os.path.join(args.output_dir, f"test_per_class_{ep_tag}.csv")
            out_raw     = os.path.join(args.output_dir, f"test_results_raw_{ep_tag}.csv")
            res_test = evaluate_loader(model, test_loader, device, class_metrics=True, save_raw=out_raw)
            write_test_metrics(res_test, out_overall, out_percls, num_classes)
            logging.info(f"Saved test CSVs: {out_overall}, {out_percls}, {out_raw}")

        # Save best by val mAP@50
        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            logging.info(f"Saved new best model (val mAP@50={best_map50:.4f})")

    logging.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--train_path', type=str, default="/sentinelkilndb_bechmarking_data/train")
    parser.add_argument('--val_path',   type=str, default="/sentinelkilndb_bechmarking_data/val")
    parser.add_argument('--test_path',  type=str, default="/sentinelkilndb_bechmarking_data/test")
    parser.add_argument('--output_dir', type=str, default="/work_dirs/panopticon_train")
    parser.add_argument('--device',     type=str, default="cuda:1")

    # model / input sizing
    parser.add_argument('--input_size', type=int, default=224)  # multiple of 14
    parser.add_argument('--num_classes', type=int, default=4)   # background + 3

    # training
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=8)   # keep small; ViT+detector is heavy
    parser.add_argument('--num_workers',  type=int,   default=8)
    parser.add_argument('--head_lr',      type=float, default=1e-4)
    parser.add_argument('--backbone_lr',  type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    # testing cadence
    parser.add_argument('--eval_test_every', type=int, default=5)

    # resume
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint_epoch_xxx.pth')

    args = parser.parse_args()
    main(args)
