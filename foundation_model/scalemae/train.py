#!/usr/bin/env python3
# train_scalemae_fixed.py
import os, sys, csv, math, argparse, logging
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchmetrics.detection import MeanAveragePrecision

# ----- TorchGeo Scale-MAE -----
try:
    from torchgeo.models import scalemae_large_patch16, ScaleMAELarge16_Weights
except ImportError:
    print("Error: torchgeo is missing. Try: pip install 'torchgeo>=0.7'")
    sys.exit(1)

# -------------------------
# Logging
# -------------------------
def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training2.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

# -------------------------
# CSV Logger
# -------------------------
class CSVLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "train_loss", "val_map", "val_map50", "lr"])

    def log(self, epoch, train_loss, val_map, val_map50, lr):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_map, val_map50, lr])

# -------------------------
# Dataset (PNG/JPG) + YOLO-OBB(9) → AABB
# IMPORTANT: no normalization here; Faster R-CNN will normalize internally.
# -------------------------
class BrickKilnDataset(Dataset):
    def __init__(self, root: str, split: str, input_size: int = 224):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.label_dir = self.root / "yolo_obb_labels"
        self.input_size = input_size

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),                      # keep [0,1], no Normalize() here
        ])

        all_files = sorted([f for f in os.listdir(self.img_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        logging.info(f"Scanning {len(all_files)} images in {self.img_dir}...")
        self.img_files = [n for n in tqdm(all_files, desc=f"Verify {split} data") if self._ok(n)]
        logging.info(f"Found {len(self.img_files)} valid images")

    def _ok(self, img_name: str) -> bool:
        p = self.label_dir / f"{Path(img_name).stem}.txt"
        if not p.exists(): return False
        with open(p, "r") as f:
            for line in f:
                if len(line.strip().split()) == 9:
                    return True
        return False

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx: int):
        name = self.img_files[idx]
        img = Image.open(self.img_dir / name).convert("RGB")
        img_t = self.transform(img)            # [3,H,W] in [0,1]
        _, h, w = img_t.shape

        boxes, labels = [], []
        with open(self.label_dir / f"{Path(name).stem}.txt", "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) != 9: continue
                cls_id = int(p[0]) + 1  # background=0
                obb = np.array([float(x) for x in p[1:]], dtype=np.float32)
                xs, ys = obb[0::2] * w, obb[1::2] * h
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        return img_t, target

def collate_fn(batch):
    batch = [b for b in batch if b[1]["boxes"].shape[0] > 0]
    if not batch: return None, None
    return tuple(zip(*batch))

# -------------------------
# Scale-MAE backbone wrapper (use _pos_embed!)
# -------------------------
class ScaleMAEBackboneWrapper(nn.Module):
    """
    ViT-L/16 encoder (1024-d). Uses Scale-MAE _pos_embed for resolution-aware PE.
    Exposes a 4-level pyramid for detection.
    """
    def __init__(self, image_size: int = 224, freeze_backbone: bool = False):
        super().__init__()
        # returns a ScaleMAE model
        self.vit = scalemae_large_patch16(weights=ScaleMAELarge16_Weights.FMOW_RGB)
        ps = getattr(self.vit.patch_embed, "patch_size", 16)
        self.patch_size = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        if image_size % self.patch_size != 0:
            raise ValueError(f"input_size must be multiple of patch size {self.patch_size}")
        self.image_size = image_size
        self.embed_dim = int(getattr(self.vit, "embed_dim", 1024))
        self.out_channels = self.embed_dim

        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

        # Tiny FPN
        self.down4 = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 2, 1)
        self.down5 = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 2, 1)
        self.down6 = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 2, 1)
        self.norm3 = nn.GroupNorm(32, self.embed_dim)
        self.norm4 = nn.GroupNorm(32, self.embed_dim)
        self.norm5 = nn.GroupNorm(32, self.embed_dim)
        self.norm6 = nn.GroupNorm(32, self.embed_dim)

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        # 1) patchify
        x = self.vit.patch_embed(x)           # [B, N, C]
        # 2) Scale-MAE’s resolution-aware positional embeddings (+ cls token & pos_drop)
        x = self.vit._pos_embed(x)            # [B, 1+N, C]
        # 3) transformer blocks + norm
        for blk in self.vit.blocks:
            x = blk(x)
        if hasattr(self.vit, "norm") and self.vit.norm is not None:
            x = self.vit.norm(x)
        # 4) drop cls → tokens
        return x[:, 1:, :]                    # [B, N, C]

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        B, C, H, W = x.shape
        assert H == W == self.image_size, f"Expected {self.image_size}x{self.image_size}, got {H}x{W}"
        tokens = self._tokens(x)                                  # [B, N, C]
        side = int(math.isqrt(tokens.shape[1]))                   # 224/16=14 → 14x14
        feat = tokens.permute(0, 2, 1).reshape(B, self.embed_dim, side, side)

        p3 = self.norm3(feat)
        p4 = self.norm4(self.down4(p3))
        p5 = self.norm5(self.down5(p4))
        p6 = self.norm6(self.down6(p5))
        return OrderedDict({"0": p3, "1": p4, "2": p5, "3": p6})

# -------------------------
# Build Faster R-CNN
#   - Only one normalization (internal to RCNN).
#   - Smaller anchors help small objects.
# -------------------------
def create_model(num_classes: int, image_size: int):
    backbone = ScaleMAEBackboneWrapper(image_size=image_size, freeze_backbone=False)

    anchor_generator = AnchorGenerator(  
        sizes=( (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )
    roi_pooler = MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=7, sampling_ratio=2)

    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=image_size,
        max_size=image_size,
        image_mean=[0.485, 0.456, 0.406],   
        image_std=[0.229, 0.224, 0.225],
    )

# -------------------------
# Train / Validate
# -------------------------
def train_one_epoch(model, optimizer, data_loader, device, scaler=None, clip_grad=1.0):
    model.train()
    total, steps = 0.0, 0
    for images, targets in tqdm(data_loader, desc="Training"):
        if images is None: continue
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total += float(loss.detach().item())
        steps += 1
    return total / max(1, steps)

@torch.no_grad()
def validate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=False)
    for images, targets in tqdm(data_loader, desc="Validation"):
        if images is None: continue
        images = [img.to(device, non_blocking=True) for img in images]
        preds = model(images)
        preds = [{k: v.to('cpu') for k, v in p.items()} for p in preds]
        t_cpu = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
        metric.update(preds, t_cpu)
    res = metric.compute()
    return float(res.get('map', torch.tensor(0.0)).item()), float(res.get('map_50', torch.tensor(0.0)).item())

# -------------------------
# Main
# -------------------------
def main(args):
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    csv_logger = CSVLogger(os.path.join(args.output_dir, "results2.csv"))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Data
    train_dataset = BrickKilnDataset(args.train_path, 'train', args.input_size)
    val_dataset   = BrickKilnDataset(args.val_path,   'val',   args.input_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model
    num_classes = 4
    model = create_model(num_classes=num_classes, image_size=args.input_size).to(device)

    # Two-group LR: tiny for ViT backbone, larger for detection heads
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (backbone_params if n.startswith("backbone.vit") else head_params).append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params,    "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_map50 = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler=scaler, clip_grad=1.0)
        val_map, val_map50 = validate(model, val_loader, device)
        scheduler.step()

        lr_disp = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch {epoch} - Loss: {train_loss:.4f} | mAP: {val_map:.4f} | mAP@50: {val_map50:.4f} | lr: {lr_disp:.6f}")
        csv_logger.log(epoch, train_loss, val_map, val_map50, lr_disp)

        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model2.pth"))
            logging.info(f"Saved new best (mAP@50={best_map50:.4f})")

    logging.info("Training finished.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_path', type=str, default="/sentinelkilndb_bechmarking_data/train")
    ap.add_argument('--val_path',   type=str, default="/sentinelkilndb_bechmarking_data/val")
    ap.add_argument('--output_dir', type=str, default="/work_dirs/scalemae_train")
    ap.add_argument('--device',     type=str, default="cuda:0")

    # Scale-MAE ViT-L/16 is built for 224; keep multiples of 16 (224 recommended)
    ap.add_argument('--input_size', type=int, default=224)

    ap.add_argument('--epochs',       type=int,   default=50)
    ap.add_argument('--batch_size',   type=int,   default=32)
    ap.add_argument('--num_workers',  type=int,   default=8)
    ap.add_argument('--head_lr',      type=float, default=1e-4)
    ap.add_argument('--backbone_lr',  type=float, default=5e-6)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--amp',          action='store_true')
    args = ap.parse_args()
    main(args)
