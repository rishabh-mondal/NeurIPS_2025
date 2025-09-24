import os
import sys
import csv
import math
import logging
import argparse
import itertools
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Tuple, Optional, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.detection import MeanAveragePrecision

# =============== Logging ===============
def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training3.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )

# =============== CSV Logger ===============
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

# =============== Dataset (PNG RGB + YOLO-OBB→AABB) ===============
class BrickKilnDataset(Dataset):
    """
    Expects:
      root/
        images/*.png
        yolo_obb_labels/*.txt   # 9-tuple per line (cls cx cy x2 y2 x3 y3 x4 y4) normalized
    """
    def __init__(self, root: str, split: str, input_size: int = 120):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images"
        self.label_dir = self.root / "yolo_obb_labels"

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),  # [0,1] RGB
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

# =============== 2D ALiBi helper (not critical for ViT fallback; kept for compatibility) ===============
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

# =============== FPN Neck ===============
class NeckFPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, 1)
        self.out3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.down4 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.out4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.down5 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.out5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.down6 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.out6 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        p3 = F.gelu(self.out3(self.lateral(x)))
        p4 = F.gelu(self.out4(self.down4(p3)))
        p5 = F.gelu(self.out5(self.down5(p4)))
        p6 = F.gelu(self.out6(self.down6(p5)))
        return {"0": p3, "1": p4, "2": p5, "3": p6}

# =============== SatMAE++-style Backbone (with robust timm fallback) ===============
class SatMAEPPBackboneWithFPN(nn.Module):
   
    def __init__(
        self,
        in_bands: int = 3,
        fpn_out: int = 256,
        timm_model_name: str = "vit_base_patch16_224",
        timm_pretrained: bool = True,
        patch_size_hint: Optional[int] = 16,
        encoder_dim_override: Optional[int] = None,
        # satmae++ (optional) – provide a callable encoder as self.encoder with forward_encoder(x, mask_ratio=0.0)
        satmaepp_encoder: Optional[nn.Module] = None,
        expect_rgb_bgr: bool = False,  # set True if your RGB weights expect BGR order
    ):
        super().__init__()
        self.in_bands = in_bands
        self.patch_size_hint = patch_size_hint
        self.expect_rgb_bgr = expect_rgb_bgr
        self.encoder = satmaepp_encoder

        if self.encoder is None:
            # Fallback: timm ViT
            import timm
            self.vit = timm.create_model(timm_model_name, pretrained=timm_pretrained)
            self.vit.eval()
            # Guess embed dim
            d = None
            for attr in ["embed_dim", "num_features", "feature_info"]:
                if hasattr(self.vit, attr) and isinstance(getattr(self.vit, attr), int):
                    d = int(getattr(self.vit, attr))
            if d is None and hasattr(self.vit, "embed_dim"):
                d = int(getattr(self.vit, "embed_dim"))
            if d is None:
                d = 768  # vit base
            self.encoder_dim = encoder_dim_override or d
        else:
            # SatMAE++ provided by user code
            d = None
            for attr in ["embed_dim", "dim", "hidden_dim"]:
                if hasattr(self.encoder, attr):
                    d = int(getattr(self.encoder, attr))
                    break
            if d is None:
                d = encoder_dim_override or 1024  # default ViT-L guess
            self.encoder_dim = d

        self.neck = NeckFPN(in_channels=self.encoder_dim, out_channels=fpn_out)
        self.out_channels = fpn_out

    @torch.no_grad()
    def _grid_hint(self, x: torch.Tensor) -> Optional[int]:
        if self.patch_size_hint is None:
            return None
        h, w = x.shape[-2:]
        if h == w and (h % self.patch_size_hint == 0):
            return h // self.patch_size_hint
        return None

    @torch.no_grad()
    def _tokens_to_grid(self, tokens: torch.Tensor, maybe_has_cls: bool, size_hint: Optional[int] = None):
        B, N, D = tokens.shape
        if maybe_has_cls and (int(math.sqrt(N - 1)) ** 2 == (N - 1)):
            tokens = tokens[:, 1:, :]
            N = N - 1
        s = int(math.sqrt(N))
        if s * s != N:
            if size_hint is not None and size_hint * size_hint == N:
                s = size_hint
            else:
                raise ValueError(f"Cannot reshape tokens into square grid: N={N}, hint={size_hint}")
        feat = tokens.transpose(1, 2).reshape(B, D, s, s).contiguous()
        return feat

    @torch.no_grad()
    def _encode_vit(self, x: torch.Tensor) -> torch.Tensor:
        # timm ViT forward_features returns either [B,N,D] or a dict; handle both
        if hasattr(self.vit, "forward_features"):
            out = self.vit.forward_features(x)
            if isinstance(out, dict):
                # common keys: 'x' or 'tokens'
                if "x" in out and out["x"].dim() == 3:
                    tokens = out["x"]
                elif "tokens" in out and out["tokens"].dim() == 3:
                    tokens = out["tokens"]
                else:
                    # last feature map (if provided as [B,C,H',W'])
                    for v in out.values():
                        if isinstance(v, torch.Tensor) and v.dim() == 4:
                            return v
                    raise RuntimeError("Unsupported timm forward_features output.")
            else:
                # Often timm returns [B,N,D] tokens
                tokens = out
            return self._tokens_to_grid(tokens, maybe_has_cls=True, size_hint=self._grid_hint(x))
        else:
            raise RuntimeError("timm ViT model lacks forward_features.")

    @torch.no_grad()
    def forward(self, x_list) -> Dict[str, torch.Tensor]:
        # torchvision passes List[Tensor]; stack to batch
        if isinstance(x_list, (list, tuple)):
            x = torch.stack(x_list, dim=0)
        else:
            x = x_list

        # Optional BGR quirk for certain RGB weights
        if self.expect_rgb_bgr and x.shape[1] == 3:
            x = x[:, [2, 1, 0], :, :].contiguous()

        if self.encoder is None:
            feat = self._encode_vit(x)   # [B, D, H', W']
        else:
            out = self.encoder.forward_encoder(x, mask_ratio=0.0)
            tokens = out[0] if isinstance(out, (tuple, list)) else out
            feat = self._tokens_to_grid(tokens, maybe_has_cls=True, size_hint=self._grid_hint(x))

        return self.neck(feat)

# =============== Detector factory ===============
def create_model(
    num_classes: int,
    image_size: int,
    in_bands: int = 3,
    fpn_out: int = 256,
    anchor_sizes: Tuple[Tuple[int, ...], ...] = ( (32,), (64,), (128,), (256,),),
    aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),) * 4,
    use_bgr_rgb_quirk: bool = False,
):
    backbone = SatMAEPPBackboneWithFPN(
        in_bands=in_bands,
        fpn_out=fpn_out,
        timm_model_name="vit_base_patch8_224",
        timm_pretrained=True,
        patch_size_hint=8,
        satmaepp_encoder=None,     # <-- plug a real SatMAE++ encoder if you have one
        expect_rgb_bgr=use_bgr_rgb_quirk,
    )

    rpn_anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,  # include background internally
        rpn_anchor_generator=rpn_anchor_gen,
        box_roi_pool=roi_pooler,
        # Lock transforms so they don't resize to 800
        min_size=image_size,
        max_size=image_size,
         image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
       
    )
    return model

# =============== Train / Validate / Test ===============
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss, steps = 0.0, 0
    for images, targets in tqdm(data_loader, desc="Training"):
        if images is None:
            continue
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
    return total_loss / max(1, steps)

@torch.no_grad()
def evaluate_map(model, data_loader, device, desc="Eval"):
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=False)
    for images, targets in tqdm(data_loader, desc=desc):
        if images is None:
            continue
        images = [img.to(device) for img in images]
        preds = model(images)
        preds = [{k: v.to('cpu') for k, v in p.items()} for p in preds]
        t_cpu = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
        metric.update(preds, t_cpu)
    res = metric.compute()
    map_ = float(res.get('map', torch.tensor(0.)))
    map50 = float(res.get('map_50', torch.tensor(0.)))
    return map_, map50

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    csv_logger = CSVLogger(os.path.join(args.output_dir, "results3.csv"))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Datasets / Loaders
    train_dataset = BrickKilnDataset(args.train_path, 'train', args.input_size)
    val_dataset   = BrickKilnDataset(args.val_path,   'val',   args.input_size)
    test_dataset  = BrickKilnDataset(args.test_path,  'test',  args.input_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model
    # Your CROMA script used 4 total classes (background + 3 kiln types) by setting labels = class_id+1.
    # FasterRCNN in torchvision expects num_classes = (# foreground classes + 1 for background).
    num_foreground = 3
    model = create_model(num_classes=num_foreground + 1, image_size=args.input_size,
                         in_bands=3, use_bgr_rgb_quirk=False).to(device)

    # Parameter groups
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # crude split: neck + rpn + roi heads as "head", encoder as "backbone"
        if "neck" in name or "rpn" in name or "roi_heads" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    logging.info(f"Transform lock: min_size={model.transform.min_size}, max_size={model.transform.max_size}")

    best_map50 = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_map, val_map50 = evaluate_map(model, val_loader, device, desc="Validation")
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch} - Loss: {train_loss:.4f} | mAP: {val_map:.4f} | mAP@50: {val_map50:.4f} | lr: {current_lr:.6f}")
        csv_logger.log(epoch, train_loss, val_map, val_map50, current_lr)

        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model3.pth"))
            logging.info(f"Saved new best (mAP@50={best_map50:.4f})")

    # Final test evaluation
    test_map, test_map50 = evaluate_map(model, test_loader, device, desc="Test")
    logging.info(f"[TEST] mAP: {test_map:.4f} | mAP@50: {test_map50:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ====== your exact defaults ======
    parser.add_argument('--train_path', type=str, default="/sentinelkilndb_bechmarking_data/train")
    parser.add_argument('--val_path',   type=str, default="/sentinelkilndb_bechmarking_data/val")
    parser.add_argument('--test_path',  type=str, default="/sentinelkilndb_bechmarking_data/test")
    parser.add_argument('--output_dir', type=str, default="/work_dirs/satmaepp_train")
    parser.add_argument('--device',     type=str, default="cuda:1")

    # Image size (square). ViT-B/16 likes multiples of 16; keep 128/144/160/176…; 120 also ok due to resize.
    parser.add_argument('--input_size', type=int, default=224)

    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=8)
    parser.add_argument('--num_workers',  type=int,   default=8)
    parser.add_argument('--head_lr',      type=float, default=1e-4)
    parser.add_argument('--backbone_lr',  type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    args = parser.parse_args()
    main(args)
