import os
import sys
import logging
import argparse
from pathlib import Path
from collections import OrderedDict
import csv

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange

# --- PyTorch Imports ---
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- TorchVision & TorchMetrics Imports ---
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchmetrics.detection import MeanAveragePrecision
import torch.nn as nn

# --- Add models to Path and Import ---
sys.path.insert(0, str(Path(__file__).parent / 'models'))
try:
    from satmae_vit import vit_large_patch16 as SatMAE_ViT
    from pos_embed import interpolate_pos_embed
except ImportError as e:
    print(f"Error: Could not import SatMAE model. Ensure 'models/satmae_vit.py' and 'models/pos_embed.py' exist. Details: {e}")
    sys.exit(1)

# --- Setup Logging ---
def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training_satmae.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

# --- Custom Dataset Class ---
class BrickKilnDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.img_dir = self.root / 'images'
        self.label_dir = self.root / 'yolo_obb_labels'
        
        self.input_size = 224
        logging.info(f"Dataset at {root} resized to {self.input_size}x{self.input_size}.")
        
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.img_files = []
        all_files = sorted(os.listdir(self.img_dir))
        for img_name in tqdm(all_files, desc=f"Verifying {root}"):
            if self._has_valid_annotations(img_name):
                self.img_files.append(img_name)
        
        logging.info(f"Found {len(self.img_files)} images with valid annotations in {self.img_dir}")

    def _has_valid_annotations(self, img_name: str) -> bool:
        label_path = self.label_dir / f"{Path(img_name).stem}.txt"
        if not label_path.exists(): 
            return False
        with open(label_path, 'r') as f:
            for line in f:
                if len(line.strip().split()) == 9:
                    return True
        return False

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_name = self.img_files[idx]
        img_path = self.img_dir / img_name
        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
        except Exception as e:
            logging.error(f"Error opening {img_path}: {e}")
            return torch.zeros(3, self.input_size, self.input_size), {"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.int64)}

        _, new_height, new_width = img_tensor.shape
        boxes, labels = [], []
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                try:
                    parts = line.strip().split()
                    class_id = int(parts[0]) + 1
                    obb = np.array([float(p) for p in parts[1:]])
                    x_coords, y_coords = obb[0::2] * new_width, obb[1::2] * new_height
                    xmin, ymin, xmax, ymax = np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id)
                except:
                    continue

        return img_tensor, {"boxes": torch.as_tensor(boxes, dtype=torch.float32), "labels": torch.as_tensor(labels, dtype=torch.int64)}

def collate_fn(batch):
    batch = [item for item in batch if item[1]['boxes'].shape[0] > 0]
    if not batch: return None, None
    return tuple(zip(*batch))

# --- SatMAE Backbone Wrapper ---
class SatMAEBackboneWrapper(nn.Module):
    def __init__(self, pretrained_path: str):
        super().__init__()
        self.input_size = 224
        logging.info("Initializing SatMAE ViT-Large backbone...")
        
        self.model = SatMAE_ViT(
            patch_size=16,
            img_size=self.input_size,
            in_chans=3,
            num_classes=1000,
            global_pool=True,
        )
        self.out_channels = self.model.embed_dim
        torch.serialization.add_safe_globals([argparse.Namespace])
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        msg = self.model.load_state_dict(checkpoint['model'], strict=False)
        logging.info(f"Loaded SatMAE weights: {msg}")
        self.body = self.model
        
    def forward(self, x: torch.Tensor) -> OrderedDict:
        patch_tokens = self.body.forward_backbone(x)
        num_patches = patch_tokens.shape[1]
        height = width = int(num_patches**0.5)
        feature_map = rearrange(patch_tokens, 'b (h w) c -> b c h w', h=height, w=width)
        return OrderedDict([("0", feature_map)])

def create_model(weights_path: str, num_classes: int) -> nn.Module:
    backbone = SatMAEBackboneWrapper(pretrained_path=weights_path)
    anchor_sizes = ((32, 64, 128, 256),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=224,
        max_size=224
    )
    return model

# --- Train and Validate ---
def train_one_epoch(model, optimizer, data_loader, device, epoch, csv_writer):
    model.train()
    loop = tqdm(data_loader, desc=f"Epoch {epoch} [Training]")
    total_loss = 0.0
    
    for images, targets in loop:
        if images is None: continue
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        if not torch.isfinite(losses):
            logging.error(f"Loss is {losses}, stopping training."); sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        loop.set_postfix(loss=losses.item())

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    logging.info(f"Epoch {epoch} Training - Average Loss: {avg_loss:.4f}")
    csv_writer.writerow([epoch, avg_loss, "NA", "NA"])  # val metrics filled later
    return avg_loss

@torch.no_grad()
def validate(model, data_loader, device, epoch, csv_writer):
    model.eval()
    loop = tqdm(data_loader, desc=f"Epoch {epoch} [Validation]")
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=False)
    
    for images, targets in loop:
        if images is None: continue
        images = [img.to(device) for img in images]
        predictions = model(images)
        predictions = [{k: v.to('cpu') for k, v in p.items()} for p in predictions]
        metric.update(predictions, targets)
            
    try:
        val_metrics = metric.compute()
        map_50 = val_metrics['map_50'].item()
        map_val = val_metrics['map'].item()
        logging.info(f"Validation Epoch {epoch} - mAP: {map_val:.4f}, mAP@50: {map_50:.4f}")
        return map_val, map_50
    except Exception as e:
        logging.error(f"Could not compute validation metrics: {e}")
        return 0.0, 0.0

# --- Main ---
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    
    logging.info(f"Starting SatMAE training run with args: {args}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    logging.info("Setting up datasets...")
    train_dataset = BrickKilnDataset(args.train_path)
    val_dataset = BrickKilnDataset(args.val_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    logging.info("Creating model...")
    model = create_model(weights_path=args.weights_path, num_classes=4).to(device)
    
    for param in model.backbone.parameters(): 
        param.requires_grad = True
    param_groups = [
        {'params': model.roi_heads.parameters(), 'lr': args.head_lr},
        {'params': model.rpn.parameters(), 'lr': args.head_lr},
        {'params': model.backbone.parameters(), 'lr': args.backbone_lr}
    ]
    optimizer = AdamW(param_groups, lr=args.head_lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.backbone_lr * 0.01)

    csv_path = os.path.join(args.output_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Epoch", "Train_Loss", "Val_mAP", "Val_mAP50"])

        best_map = 0.0
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1, csv_writer)
            lr_scheduler.step()
            map_val, map_50 = validate(model, val_loader, device, epoch + 1, csv_writer)
            
            # Overwrite last row with val metrics
            f.flush()

            if map_50 > best_map:
                best_map = map_50
                best_model_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                logging.info(f" New best model saved to {best_model_path} with mAP@50: {best_map:.4f}")

            checkpoint_path = os.path.join(args.output_dir, 'last_checkpoint.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_map': best_map,
                'args': args
            }
            torch.save(checkpoint, checkpoint_path)

    logging.info(f"--- Training Finished. Best mAP@50: {best_map:.4f} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a brick kiln detector with SatMAE backbone.")
    parser.add_argument('--train_path', type=str, default='/sentinelkilndb_bechmarking_data/train')
    parser.add_argument('--val_path', type=str, default='/sentinelkilndb_bechmarking_data/val')
    parser.add_argument('--test_path', type=str, default='/sentinelkilndb_bechmarking_data/test')
    parser.add_argument('--weights_path', type=str, default='pre_trained_weights/satmae/fmow_pretrain.pth')
    parser.add_argument('--output_dir', type=str, default='work_dirs')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    
    args = parser.parse_args()
    main(args)
