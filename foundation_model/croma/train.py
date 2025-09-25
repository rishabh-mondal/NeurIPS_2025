import os
import sys
import csv
import math
import itertools
import logging
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
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
    def __init__(self, root: str, split: str, input_size: int = 120):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images"
        self.label_dir = self.root / "yolo_obb_labels"

        # Keep as [0,1], no ImageNet normalization (works better with learned 1x1 RGB->12 adapter)
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
        }
        return img_tensor, target


def collate_fn(batch):
    batch = [item for item in batch if item[1]["boxes"].shape[0] > 0]
    if not batch:
        return None, None
    return tuple(zip(*batch))


# -------------------------
# 2D ALiBi (recompute to match current patch grid)
# -------------------------
def _get_2dalibi(num_heads: int, num_patches: int) -> torch.Tensor:
    # inspired by TorchGeo CROMA implementation
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
# CROMA Backbone Wrapper (optical-only, RGB->12ch)
# -------------------------
class CromaBackboneWrapper(nn.Module):
    """
    - Adapts RGB(3) -> pseudo S2(12) via 1x1 conv
    - Uses croma_base(weights=CROMABase_Weights.CROMA_VIT, modalities=['optical'], image_size=<input_size>)
    - Reshapes tokens to [B,C,H',W'] where H'=W'=H/8
    - Builds simple pyramid P3..P6, returns dict { '0','1','2','3' }
    - Recomputes relative-position bias when the patch grid changes (robust to any multiple-of-8 size)
    """
    def __init__(self, image_size: int = 120, freeze_croma: bool = False):
        super().__init__()
        assert image_size % 8 == 0, "CROMA expects image_size to be a multiple of 8"

        self.image_size = image_size

        # RGB -> 12 bands adapter
        self.rgb_to_ms = nn.Conv2d(3, 12, kernel_size=1, bias=True)

        # Load CROMA (optical only)
        self.croma = croma_base(
            weights=CROMABase_Weights.CROMA_VIT,
            modalities=['optical'],
            image_size=image_size,
        )

        if freeze_croma:
            for p in self.croma.parameters():
                p.requires_grad = False

        # CROMA base config
        self.encoder_dim = self.croma.encoder_dim  # 768
        self.patch_size = self.croma.patch_size    # 8
        self.num_heads = self.croma.num_heads      # 16

        # Tell torchvision how many channels each feature map has
        self.out_channels = self.encoder_dim  # <-- REQUIRED by torchvision detectors

        # Simple FPN-ish downsamples (keep channel dim constant)
        self.down4 = nn.Conv2d(self.encoder_dim, self.encoder_dim, kernel_size=3, stride=2, padding=1)
        self.down5 = nn.Conv2d(self.encoder_dim, self.encoder_dim, kernel_size=3, stride=2, padding=1)
        self.down6 = nn.Conv2d(self.encoder_dim, self.encoder_dim, kernel_size=3, stride=2, padding=1)

        self.norm3 = nn.GroupNorm(32, self.encoder_dim)
        self.norm4 = nn.GroupNorm(32, self.encoder_dim)
        self.norm5 = nn.GroupNorm(32, self.encoder_dim)
        self.norm6 = nn.GroupNorm(32, self.encoder_dim)

    def _ensure_attn_bias(self, H: int, device: torch.device):
        # H is input image height (== width). Patch grid side = H/8.
        h_tokens = H // self.patch_size
        num_patches = h_tokens * h_tokens
        current = getattr(self.croma, "attn_bias", None)
        if (current is None) or (current.shape[-1] != num_patches):
            new_bias = _get_2dalibi(self.num_heads, num_patches).to(device)
            # TorchGeo CROMA reads self.attn_bias in forward(); override safely
            self.croma.attn_bias = new_bias

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        # x: [B,3,H,W] (H=W and multiple of 8 guaranteed by our transform + RCNN min/max)
        B, C, H, W = x.shape
        assert H == W, "CROMA expects square inputs"
        self._ensure_attn_bias(H, x.device)

        x12 = self.rgb_to_ms(x)                      # [B,12,H,W]
        out = self.croma(x_optical=x12)              # dict with 'optical_encodings': [B,N,C]
        tokens = out["optical_encodings"]
        B, N, Cdim = tokens.shape
        side = int(math.isqrt(N))
        feat = tokens.permute(0, 2, 1).reshape(B, Cdim, side, side)  # [B,768,H/8,W/8]

        p3 = self.norm3(feat)            # stride ~8
        p4 = self.norm4(self.down4(p3))  # ~16
        p5 = self.norm5(self.down5(p4))  # ~32
        p6 = self.norm6(self.down6(p5))  # ~64

        return OrderedDict({"0": p3, "1": p4, "2": p5, "3": p6})


# -------------------------
# Build Faster R-CNN (lock size, neutral mean/std)
# -------------------------
def create_model(num_classes: int, image_size: int):
    backbone = CromaBackboneWrapper(image_size=image_size, freeze_croma=False)

    # Anchor sizes chosen for ~30 px objects at 120x120 (10 m/px, ~300 m kilns)
    anchor_generator = AnchorGenerator(
        sizes=( (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # IMPORTANT: prevent torchvision from resizing to 800
        min_size=image_size,
        max_size=image_size,
        image_mean=[0.485, 0.456, 0.406], 
        image_std=[0.229, 0.224, 0.225]
    )
    return model


# -------------------------
# Train / Validate
# -------------------------
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    steps = 0
    for images, targets in tqdm(data_loader, desc="Training"):
        if images is None:
            continue
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        steps += 1

    return total_loss / max(1, steps)


@torch.no_grad()
def validate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=False)
    for images, targets in tqdm(data_loader, desc="Validation"):
        if images is None:
            continue
        images = [img.to(device) for img in images]
        preds = model(images)
        preds = [{k: v.to('cpu') for k, v in p.items()} for p in preds]
        t_cpu = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
        metric.update(preds, t_cpu)
    res = metric.compute()
    return res.get('map', torch.tensor(0.)).item(), res.get('map_50', torch.tensor(0.)).item()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    csv_logger = CSVLogger(os.path.join(args.output_dir, "results.csv"))

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
    num_classes = 4  # background + 3 kiln classes
    model = create_model(num_classes=num_classes, image_size=args.input_size).to(device)

    # Two LR groups: smaller LR for pretrained CROMA; larger for adapter + heads
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "croma" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Log transform lock
    logging.info(f"RCNN transform min_size={model.transform.min_size}, max_size={model.transform.max_size}")

    best_map50 = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_map, val_map50 = validate(model, val_loader, device)
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch} - Loss: {train_loss:.4f}, mAP: {val_map:.4f}, mAP@50: {val_map50:.4f} - lr: {current_lr:.6f}")
        csv_logger.log(epoch, train_loss, val_map, val_map50, current_lr)

        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            logging.info(f"Saved new best model (mAP@50={best_map50:.4f})")

    logging.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default="/sentinelkilndb_bechmarking_data/train")
    parser.add_argument('--val_path',   type=str, default="/sentinelkilndb_bechmarking_data/val")
    parser.add_argument('--test_path',  type=str, default="/sentinelkilndb_bechmarking_data/test")
    parser.add_argument('--output_dir', type=str, default="/work_dirs/croma_train")
    parser.add_argument('--device',     type=str, default="cuda:1")

    # Pick any multiple of 8 (120 recommended for CROMA base). Everything else adapts.
    parser.add_argument('--input_size', type=int, default=120)

    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=8)
    parser.add_argument('--num_workers',  type=int,   default=8)
    parser.add_argument('--head_lr',      type=float, default=1e-4)
    parser.add_argument('--backbone_lr',  type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    args = parser.parse_args()
    main(args)
