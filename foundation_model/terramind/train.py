import os
import sys
import csv
import logging
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchmetrics.detection import MeanAveragePrecision
import torch.nn as nn

try:
    from terratorch.registry import BACKBONE_REGISTRY
except ImportError:
    print("Error: Could not import terratorch. Please ensure it is installed correctly.")
    sys.exit(1)


# --- Setup Logging ---
def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# --- CSV Logger ---
class CSVLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_map", "val_map50", "lr"])

    def log(self, epoch, train_loss, val_map, val_map50, lr):
        # Ensure numeric values, fallback to NaN if None
        val_map = float('nan') if val_map is None else val_map
        val_map50 = float('nan') if val_map50 is None else val_map50
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_map, val_map50, lr])



# --- Dataset ---
class BrickKilnDataset(Dataset):
    def __init__(self, root: str, split: str, input_size: int = 224):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images"
        self.label_dir = self.root / "yolo_obb_labels"

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.img_files = []
        all_files = sorted(os.listdir(self.img_dir))
        logging.info(f"Scanning {len(all_files)} images in {self.img_dir}...")
        for img_name in tqdm(all_files, desc=f"Verifying {split} data"):
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
                class_id = int(parts[0]) + 1
                obb = np.array([float(p) for p in parts[1:]])
                xs, ys = obb[0::2] * w, obb[1::2] * h
                xmin, ymin, xmax, ymax = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)

        return img_tensor, {"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                            "labels": torch.as_tensor(labels, dtype=torch.int64)}


def collate_fn(batch):
    batch = [item for item in batch if item[1]['boxes'].shape[0] > 0]
    if not batch: return None, None
    return tuple(zip(*batch))


# --- TerraMind Backbone ---
class TerraMindBackboneWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BACKBONE_REGISTRY.build(
            "terramind_v1_base",
            modalities=["S2L2A"],
            pretrained=True,
            bands={"S2L2A": ["B4", "B3", "B2"]}
        )
        self.out_channels = 768
        self.return_layers_indices = [3, 5, 8, 11]

    def forward(self, x):
        features_list = self.backbone({"S2L2A": x})
        out = OrderedDict()
        for i, layer_idx in enumerate(self.return_layers_indices):
            tokens = features_list[layer_idx]
            h_w = int(np.sqrt(tokens.shape[1]))
            feat = tokens.permute(0, 2, 1).reshape(x.shape[0], self.out_channels, h_w, h_w)
            out[str(i)] = feat
        return out


def create_model(num_classes: int):
    backbone = TerraMindBackboneWrapper()
    anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,)),
                                       aspect_ratios=((0.5, 1.0, 2.0),) * 4)
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
    return FasterRCNN(backbone, num_classes=num_classes,
                      rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)


# --- Training / Validation ---
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(data_loader, desc="Training"):
        if images is None: continue
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / len(data_loader)


@torch.no_grad()
def validate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=False)
    for images, targets in tqdm(data_loader, desc="Validation"):
        if images is None: continue
        images = [img.to(device) for img in images]
        preds = model(images)
        preds = [{k: v.to('cpu') for k, v in p.items()} for p in preds]
        t_cpu = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
        metric.update(preds, t_cpu)
    res = metric.compute()
    return res['map'].item(), res['map_50'].item()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    csv_logger = CSVLogger(os.path.join(args.output_dir, "results.csv"))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    train_dataset = BrickKilnDataset(args.train_path, 'train', args.input_size)
    val_dataset = BrickKilnDataset(args.val_path, 'val', args.input_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    model = create_model(num_classes=4).to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.head_lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_map = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_map, val_map50 = validate(model, val_loader, device)
        lr_scheduler.step()

        logging.info(f"Epoch {epoch} - Loss: {train_loss:.4f}, mAP: {val_map:.4f}, mAP@50: {val_map50:.4f}")
        csv_logger.log(epoch, train_loss, val_map, val_map50, optimizer.param_groups[0]['lr'])

        if val_map50 > best_map:
            best_map = val_map50
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

    logging.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default="/sentinelkilndb_bechmarking_data/train")
    parser.add_argument('--val_path', type=str, default="/sentinelkilndb_bechmarking_data/val")
    parser.add_argument('--test_path', type=str, default="/sentinelkilndb_bechmarking_data/test")
    parser.add_argument('--output_dir', type=str, default="/work_dirs/terramind_train")
    parser.add_argument('--device', type=str, default="cuda:3")
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    args = parser.parse_args()
    main(args)
