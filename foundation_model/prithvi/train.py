import os
import sys
import csv
import math
import yaml
import argparse
import logging
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# torchmetrics mAP
try:
    from torchmetrics.detection import MeanAveragePrecision
    _TM_DET_OK = True
except Exception:
    _TM_DET_OK = False

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm


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
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_map", "val_map50", "lr"])

    def log(self, epoch, train_loss, val_map, val_map50, lr):
        val_map = float("nan") if val_map is None else float(val_map)
        val_map50 = float("nan") if val_map50 is None else float(val_map50)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_map, val_map50, lr])


# -------------------------
# Dataset: PNG RGB + YOLO-OBB (9-tuple) -> AABB
# -------------------------
class BrickKilnDataset(Dataset):
    """
    root/
      images/*.png
      yolo_obb_labels/*.txt   (each line: cls x1 y1 x2 y2 x3 y3 x4 y4) in YOLO-OBB normalized coords
    Converts OBB to AABB for Faster R-CNN training/eval.
    """
    def __init__(self, root: str, split: str, input_size: int):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images"
        self.label_dir = self.root / "yolo_obb_labels"
        self.input_size = input_size

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),   # stay in [0,1]; detector handles normalization
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
        with open(label_path, "r") as f:
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
        with open(label_path, "r") as f:
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
# Prithvi (local) Backbone Wrapper
# -------------------------
class PrithviBackboneWrapper(nn.Module):
    """
    Loads local Prithvi from a folder (with prithvi_mae.py + *.pt) and exposes a feature map pyramid.

    - Reads model_args from config.yaml (img_size, embed_dim, depth, patch_size, etc.)
    - Builds PrithviMAE(encoder_only=True) and loads .pt (ignoring pos_embed mismatch)
    - Adds RGB->6ch 1x1 adapter (so we can reuse the 6-band encoder weights)
    - Uses encoder.forward_features(...) + prepare_features_for_image_model(...) to get [B,D,H',W']
    - Builds P3..P6 via strided convs; returns OrderedDict("0","1","2","3")
    """
    def __init__(self, prithvi_dir: str, ckpt_name: str, image_size: int, freeze_backbone: bool = False):
        super().__init__()
        self.image_size = image_size

        # Add repo folder to path and import
        prithvi_dir = os.path.abspath(prithvi_dir)
        if prithvi_dir not in sys.path:
            sys.path.insert(0, prithvi_dir)
        try:
            from prithvi_100m.prithvi_mae import PrithviMAE
        except ImportError:
            raise RuntimeError(f"Could not import prithvi_mae from {prithvi_dir}")

        # Load config.yaml
        cfg_path = os.path.join(prithvi_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config.yaml in {prithvi_dir}")
        with open(cfg_path, "r") as f:
            y = yaml.safe_load(f)

        margs = y.get("model_args", {})
        # force our training sizes (T=1, keep 6 bands to match weights)
        margs.update({
            "img_size": image_size,
            "num_frames": 1,
            "in_chans": 6,
            "encoder_only": True,
        })

        # Build encoder-only model
        self.prithvi = PrithviMAE(**margs)

        # Load checkpoint (.pt)
        ckpt_path = os.path.join(prithvi_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        device = torch.device("cpu")
        state_dict = torch.load(ckpt_path, map_location=device)
        # drop pos_embed (and any mismatch) -> strict=False
        for k in list(state_dict.keys()):
            if "pos_embed" in k:
                del state_dict[k]
        missing, unexpected = self.prithvi.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded Prithvi weights. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

        # dims
        self.hidden_dim = margs.get("embed_dim", 768)
        self.patch_size = margs.get("patch_size", 16)

        # RGB->6 band adapter (learned)
        self.adapter = nn.Conv2d(3, 6, kernel_size=1, bias=True)

        if freeze_backbone:
            for p in self.prithvi.parameters():
                p.requires_grad = False
            for p in self.adapter.parameters():
                p.requires_grad = True

        # Simple FPN-ish neck
        self.out_channels = self.hidden_dim
        self.down4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 2, 1)
        self.down5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 2, 1)
        self.down6 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 2, 1)

        self.norm3 = nn.GroupNorm(32, self.hidden_dim)
        self.norm4 = nn.GroupNorm(32, self.hidden_dim)
        self.norm5 = nn.GroupNorm(32, self.hidden_dim)
        self.norm6 = nn.GroupNorm(32, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        # x: [B,3,H,W] -> adapter -> [B,6,H,W] -> add time dim: [B,6,1,H,W]
        B, C, H, W = x.shape
        assert H == W == self.image_size, f"Input must be square {self.image_size}"
        assert H % self.patch_size == 0, "Input not divisible by patch size"
        x6 = self.adapter(x)
        x5d = x6.unsqueeze(2)  # T = 1

        with torch.cuda.amp.autocast(enabled=torch.is_autocast_enabled()):
            feats_seq = self.prithvi.encoder.forward_features(x5d)  # list of [B, 1+L, D]
            maps = self.prithvi.encoder.prepare_features_for_image_model(feats_seq)  # list of [B, D*T, h, w]
            feat = maps[-1]  # last block feature, [B, D, H', W'] since T=1

            p3 = self.norm3(feat)            # stride ~patch
            p4 = self.norm4(self.down4(p3))  # ~2*patch
            p5 = self.norm5(self.down5(p4))  # ~4*patch
            p6 = self.norm6(self.down6(p5))  # ~8*patch

        return OrderedDict({"0": p3, "1": p4, "2": p5, "3": p6})


# -------------------------
# Build Faster R-CNN
# -------------------------
def create_model(num_classes: int, image_size: int, prithvi_dir: str, ckpt_name: str):
    backbone = PrithviBackboneWrapper(
        prithvi_dir=prithvi_dir,
        ckpt_name=ckpt_name,
        image_size=image_size,
        freeze_backbone=False,
    )

    anchor_generator = AnchorGenerator(
        sizes=( (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

    # neutral normalization; adapter/backbone learn statistics end-to-end
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=image_size,
        max_size=image_size,
       image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )
    return model


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


def find_latest_checkpoint(dir_path: str) -> str | None:
    if not os.path.isdir(dir_path):
        return None
    cks = [f for f in os.listdir(dir_path) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if not cks:
        return None
    def _ep(x):
        try:
            return int(x.replace("checkpoint_epoch_", "").replace(".pth", ""))
        except Exception:
            return -1
    cks.sort(key=_ep)
    return os.path.join(dir_path, cks[-1])


# -------------------------
# Train / Validate / Test
# -------------------------
def train_one_epoch(model, optimizer, data_loader, device, scaler=None):
    model.train()
    total_loss = 0.0
    steps = 0
    for images, targets in tqdm(data_loader, desc="Training"):
        if images is None:
            continue
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        total_loss += float(losses.detach().item())
        steps += 1

    return total_loss / max(1, steps)


@torch.no_grad()
def evaluate_map(model, data_loader, device, class_metrics=False):
    if not _TM_DET_OK:
        logging.warning("torchmetrics detection backend missing. Install `torchmetrics[detection]` and `pycocotools`.")
        return {"map": torch.tensor(0.0), "map_50": torch.tensor(0.0)}

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=class_metrics)
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        if images is None:
            continue
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)
        outputs = [{k: v.to("cpu") for k, v in o.items()} for o in outputs]
        targets_cpu = [{k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
        metric.update(outputs, targets_cpu)
    return metric.compute()


@torch.no_grad()
def save_test_predictions(model, data_loader, device, out_csv):
    model.eval()
    rows = []
    for images, targets in tqdm(data_loader, desc="Test-Predict"):
        if images is None:
            continue
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)
        for t, out in zip(targets, outputs):
            img_name = t.get("image_name", "unknown.png")
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            for b, s, l in zip(boxes, scores, labels):
                rows.append([img_name, float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), int(l)])

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows, columns=["image","x1","y1","x2","y2","score","label"]).to_csv(out_csv, index=False)


def write_test_metrics(res_dict, out_overall_csv, out_perclass_csv, num_classes):
    os.makedirs(os.path.dirname(out_overall_csv), exist_ok=True)
    overall = {
        "map": float(res_dict.get("map", torch.tensor(0.)).item()),
        "map_50": float(res_dict.get("map_50", torch.tensor(0.)).item()),
        "map_75": float(res_dict.get("map_75", torch.tensor(0.)).item()),
    }
    pd.DataFrame([overall]).to_csv(out_overall_csv, index=False)

    # per-class
    if "map_per_class" in res_dict and res_dict["map_per_class"] is not None:
        m = res_dict["map_per_class"].detach().cpu().numpy()
        m50 = res_dict.get("map_50_per_class", torch.zeros(num_classes)).detach().cpu().numpy()
        rows = []
        for cls in range(len(m)):
            rows.append({"class_id": int(cls+1), "map": float(m[cls]), "map_50": float(m50[cls])})
        pd.DataFrame(rows).to_csv(out_perclass_csv, index=False)


def main(args):
    # sanitize device if necessary
    if args.device.startswith("cuda") and torch.cuda.is_available():
        if ":" in args.device:
            want = int(args.device.split(":")[1])
            n = torch.cuda.device_count()
            if want >= n:
                logging.warning(f"Requested {args.device} but only {n} visible device(s). Using cuda:0.")
                args.device = "cuda:0"

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # data
    train_dataset = BrickKilnDataset(args.train_path, "train", args.input_size)
    val_dataset   = BrickKilnDataset(args.val_path,   "val",   args.input_size)
    test_dataset  = BrickKilnDataset(args.test_path,  "test",  args.input_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # model
    num_classes = args.num_classes
    model = create_model(num_classes=num_classes, image_size=args.input_size,
                         prithvi_dir=args.prithvi_dir, ckpt_name=args.ckpt_name).to(device)

    # AMP + optimizer
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    # Param groups: smaller LR for Prithvi encoder; larger for adapter + detection heads
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone.prithvi"):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    logging.info(f"RCNN transform min_size={model.transform.min_size}, max_size={model.transform.max_size}")

    # resume?
    start_epoch = 1
    best_map50 = 0.0
    resume_path = None
    if args.resume and os.path.isfile(args.resume):
        resume_path = args.resume
    elif args.auto_resume:
        cand = find_latest_checkpoint(args.output_dir)
        if cand:
            resume_path = cand
    if resume_path:
        logging.info(f"Resuming from checkpoint: {resume_path}")
        start_epoch, best_map50 = load_checkpoint(resume_path, model, optimizer, lr_scheduler, device)
        logging.info(f"Resumed at epoch={start_epoch}, best_map50={best_map50:.4f}")

    # training loop
    csv_logger = CSVLogger(os.path.join(args.output_dir, "results.csv"))
    for epoch in range(start_epoch, args.epochs + 1):
        try:
            train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
            res_val = evaluate_map(model, val_loader, device, class_metrics=False)
            val_map = float(res_val.get("map", torch.tensor(0.)).item())
            val_map50 = float(res_val.get("map_50", torch.tensor(0.)).item())
            lr_scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(f"Epoch {epoch} - Loss: {train_loss:.4f}, mAP: {val_map:.4f}, mAP@50: {val_map50:.4f} - lr: {current_lr:.6f}")
            csv_logger.log(epoch, train_loss, val_map, val_map50, current_lr)

            # save rolling checkpoint
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            save_checkpoint(ckpt_path, model, optimizer, lr_scheduler, epoch, best_map50)

            # test every N epochs
            if epoch % args.eval_test_every == 0 or epoch == args.epochs:
                ep_tag = f"epoch{epoch:03d}"
                out_overall = os.path.join(args.output_dir, f"test_overall_{ep_tag}.csv")
                out_percls  = os.path.join(args.output_dir, f"test_per_class_{ep_tag}.csv")
                out_raw     = os.path.join(args.output_dir, f"test_results_raw_{ep_tag}.csv")
                res_test = evaluate_map(model, test_loader, device, class_metrics=True)
                write_test_metrics(res_test, out_overall, out_percls, num_classes)
                save_test_predictions(model, test_loader, device, out_raw)
                logging.info(f"Saved test CSVs: {out_overall}, {out_percls}, {out_raw}")

            # best by val mAP@50
            if val_map50 > best_map50:
                best_map50 = val_map50
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
                logging.info(f"Saved new best model (val mAP@50={best_map50:.4f})")

        except Exception as e:
            # save crash checkpoint to allow resuming
            crash_path = os.path.join(args.output_dir, f"CRASH_epoch_{epoch:03d}.pth")
            try:
                save_checkpoint(crash_path, model, optimizer, lr_scheduler, epoch, best_map50)
                logging.error(f"Exception at epoch {epoch}. Saved crash checkpoint: {crash_path}")
            except Exception as ee:
                logging.error(f"Failed to save crash checkpoint: {ee}")
            raise e

    logging.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--train_path", type=str, default="/sentinelkilndb_bechmarking_data/train")
    parser.add_argument("--val_path",   type=str, default="/sentinelkilndb_bechmarking_data/val")
    parser.add_argument("--test_path",  type=str, default="/sentinelkilndb_bechmarking_data/test")
    parser.add_argument("--output_dir", type=str, default="/work_dirs/prithvi_train")

    # backbone + sizes
    parser.add_argument("--prithvi_dir", type=str, default="/prithvi_100m")
    parser.add_argument("--ckpt_name",   type=str, default="Prithvi_100M.pt")   # or Prithvi_EO_V1_100M.pt
    parser.add_argument("--input_size",  type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=4)

    # training
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=4)   # start small; raise if memory allows
    parser.add_argument("--num_workers",   type=int,   default=8)
    parser.add_argument("--head_lr",       type=float, default=1e-4)
    parser.add_argument("--backbone_lr",   type=float, default=1e-5)
    parser.add_argument("--weight_decay",  type=float, default=0.05)
    parser.add_argument("--device",        type=str,   default="cuda:1")
    parser.add_argument("--amp",           action="store_true", help="Enable mixed precision")

    # eval cadence + resume
    parser.add_argument("--eval_test_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint_epoch_xxx.pth")
    parser.add_argument("--auto_resume", action="store_true", help="pick latest checkpoint in output_dir automatically")

    args = parser.parse_args()
    main(args)
