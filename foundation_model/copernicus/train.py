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
    from torchgeo.models import CopernicusFM, copernicusfm_base, CopernicusFM_Base_Weights
    print("✅ Successfully imported CopernicusFM from TorchGeo!")
except ImportError as e:
    print(f"❌ Error importing CopernicusFM: {e}")
    print("Please ensure torchgeo is installed correctly.")
    sys.exit(1)

# --- Setup Logging ---
def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'copernicus_training2.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("🚀 CopernicusFM Object Detection Training Started")

# --- CSV Logger ---
class CSVLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_map", "val_map50", "lr", "backbone_info"])

    def log(self, epoch, train_loss, val_map, val_map50, lr, backbone_info="CopernicusFM"):
        val_map = float('nan') if val_map is None else val_map
        val_map50 = float('nan') if val_map50 is None else val_map50
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_map, val_map50, lr, backbone_info])

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

# --- DEBUG: Check what CopernicusFM actually outputs ---
def debug_copernicus_output():
    """Debug function to understand CopernicusFM output"""
    print("🔍 Debugging CopernicusFM output...")
    
    # Create a dummy model
    model = copernicusfm_base(weights=CopernicusFM_Base_Weights.CopernicusFM_ViT)
    
    # Test input
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_metadata = torch.full((2, 4), float('nan'))
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Test different forward methods
    with torch.no_grad():
        # Test forward_features
        features = model.forward_features(
            dummy_input, 
            dummy_metadata,
            wavelengths=[640, 550, 470],
            bandwidths=[100, 90, 80],
            input_mode='spectral'
        )
        print(f"forward_features output shape: {features.shape}")
        
        # Test full forward
        output = model(
            dummy_input, 
            dummy_metadata,
            wavelengths=[640, 550, 470],
            bandwidths=[100, 90, 80],
            input_mode='spectral'
        )
        print(f"full forward output shape: {output.shape}")
    
    return features.shape

# --- CORRECTED CopernicusFM Backbone ---
class CopernicusFMBackboneWrapper(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        if pretrained:
            print("📥 Loading pre-trained CopernicusFM weights...")
            self.model = copernicusfm_base(weights=CopernicusFM_Base_Weights.CopernicusFM_ViT)
        else:
            print("🆕 Initializing CopernicusFM from scratch...")
            self.model = CopernicusFM()
        
        # Freeze the backbone initially (we'll unfreeze later if needed)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get the actual output dimension by testing
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_metadata = torch.full((1, 4), float('nan'))
        
        with torch.no_grad():
            # Use the internal patch embedding directly to get spatial features
            if hasattr(self.model, 'patch_embed_spectral'):
                # Extract patch embeddings directly
                wvs = torch.tensor([640, 550, 470], device='cpu').float()
                bws = torch.tensor([100, 90, 80], device='cpu').float()
                
                patch_embeddings = self.model.patch_embed_spectral(
                    dummy_input, 
                    wavelengths=wvs, 
                    bandwidths=bws
                )
                print(f"Patch embeddings shape: {patch_embeddings.shape}")
                
                # This should give us [B, num_patches, embed_dim]
                self.out_channels = patch_embeddings.shape[-1]
                self.num_patches = patch_embeddings.shape[1]
                
            else:
                # Fallback: use forward_features
                features = self.model.forward_features(
                    dummy_input, 
                    dummy_metadata,
                    wavelengths=[640, 550, 470],
                    bandwidths=[100, 90, 80],
                    input_mode='spectral'
                )
                self.out_channels = features.shape[-1]
                print(f"Features shape: {features.shape}")
        
        print(f"✅ CopernicusFM output channels: {self.out_channels}")
        
        # Define RGB wavelengths and bandwidths
        self.wavelengths = [640, 550, 470]
        self.bandwidths = [100, 90, 80]
        
        # For spatial feature extraction, we need to modify the approach
        # Let's create a simple CNN head to convert ViT features to spatial maps
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(self.out_channels, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 768, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.adapter_out_channels = 768

    def forward(self, x):
        batch_size = x.shape[0]
        metadata = torch.full((batch_size, 4), float('nan'), device=x.device)
        
        try:
            # Method 1: Try to get patch embeddings directly
            if hasattr(self.model, 'patch_embed_spectral'):
                wvs = torch.tensor(self.wavelengths, device=x.device).float()
                bws = torch.tensor(self.bandwidths, device=x.device).float()
                
                # Get patch embeddings (should be [B, num_patches, D])
                patch_embeds = self.model.patch_embed_spectral(x, wavelengths=wvs, bandwidths=bws)
                
                # Reshape to spatial format [B, D, H, W]
                grid_size = int(np.sqrt(patch_embeds.shape[1]))
                spatial_features = patch_embeds.transpose(1, 2).reshape(
                    batch_size, self.out_channels, grid_size, grid_size
                )
                
            else:
                # Method 2: Use forward_features and create spatial representation
                features = self.model.forward_features(
                    x, metadata, 
                    wavelengths=self.wavelengths,
                    bandwidths=self.bandwidths,
                    input_mode='spectral'
                )
                
                # If we get global features, create spatial representation
                if features.dim() == 2:  # [B, D]
                    # Create a basic spatial feature map
                    grid_size = 14  # Default for 224x224 with patch size 16
                    spatial_features = features.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
                    spatial_features = spatial_features.expand(-1, -1, grid_size, grid_size)
                else:
                    # Assume [B, num_patches, D]
                    grid_size = int(np.sqrt(features.shape[1]))
                    spatial_features = features.transpose(1, 2).reshape(
                        batch_size, self.out_channels, grid_size, grid_size
                    )
            
            # Adapt features for detection
            adapted_features = self.feature_adapter(spatial_features)
            
            # Create multi-scale feature pyramid
            out = OrderedDict()
            for i in range(4):
                scale_factor = 2 ** i
                target_size = (adapted_features.shape[2] // scale_factor, 
                              adapted_features.shape[3] // scale_factor)
                out[str(i)] = nn.functional.interpolate(
                    adapted_features, size=target_size, mode='bilinear', align_corners=False
                )
            
            return out
            
        except Exception as e:
            print(f"❌ Error in CopernicusFM backbone: {e}")
            # Fallback: return random features (for debugging)
            print("Using fallback features...")
            out = OrderedDict()
            for i in range(4):
                size = 56 // (2 ** i)
                out[str(i)] = torch.randn(batch_size, self.adapter_out_channels, size, size, device=x.device)
            return out


def create_model(num_classes: int, pretrained: bool = True):
    backbone = CopernicusFMBackboneWrapper(pretrained=pretrained)
    
    # Use the adapter output channels
    print(f"Backbone output channels: {backbone.adapter_out_channels}")
    
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'], 
        output_size=7, 
        sampling_ratio=2
    )
    
    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
       
        min_size=224,
        max_size=224,
        image_mean=[0.485, 0.456, 0.406], 
        image_std=[0.229, 0.224, 0.225]
    )


# --- Training / Validation ---
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)
    processed_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} Training")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        if images is None: 
            continue
            
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        try:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Check if we have valid losses
            if losses.item() > 0:
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += losses.item()
                processed_batches += 1
                
                pbar.set_postfix({'loss': losses.item()})
            else:
                print(f"⚠️  Zero loss in batch {batch_idx}, skipping update")
                
        except Exception as e:
            print(f"❌ Error in training batch {batch_idx}: {e}")
            continue
            
    return total_loss / processed_batches if processed_batches > 0 else 0.0


@torch.no_grad()
def validate(model, data_loader, device, epoch):
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=False)
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} Validation")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        if images is None: 
            continue
            
        images = [img.to(device) for img in images]
        
        try:
            preds = model(images)
            preds = [{k: v.to('cpu') for k, v in p.items()} for p in preds]
            t_cpu = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
            metric.update(preds, t_cpu)
            
        except Exception as e:
            print(f"❌ Error in validation batch {batch_idx}: {e}")
            continue
            
    try:
        res = metric.compute()
        return res['map'].item(), res['map_50'].item()
    except:
        return 0.0, 0.0  # Return 0 instead of -1 for failed validation


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    csv_logger = CSVLogger(os.path.join(args.output_dir, "copernicus_results.csv"))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # First debug the CopernicusFM output
    debug_copernicus_output()

    train_dataset = BrickKilnDataset(args.train_path, 'train', args.input_size)
    val_dataset = BrickKilnDataset(args.val_path, 'val', args.input_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    model = create_model(num_classes=4, pretrained=True).to(device)
    
    # Separate learning rates for backbone and heads
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name and param.requires_grad:
            backbone_params.append(param)
        elif param.requires_grad:
            head_params.append(param)
    
    print(f"Backbone parameters: {len(backbone_params)}")
    print(f"Head parameters: {len(head_params)}")
    
    if len(backbone_params) == 0:
        print("⚠️  No backbone parameters to train - unfreezing backbone")
        # Unfreeze the backbone
        for param in model.backbone.parameters():
            param.requires_grad = True
        backbone_params = list(model.backbone.parameters())
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': head_params, 'lr': args.head_lr}
    ], weight_decay=args.weight_decay)
    
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_map = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_map, val_map50 = validate(model, val_loader, device, epoch)
        lr_scheduler.step()

        logging.info(f"Epoch {epoch} - Loss: {train_loss:.4f}, mAP: {val_map:.4f}, mAP@50: {val_map50:.4f}")
        csv_logger.log(epoch, train_loss, val_map, val_map50, optimizer.param_groups[0]['lr'], "CopernicusFM")

        if val_map50 > best_map:
            best_map = val_map50
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_copernicus_model.pth"))
            logging.info(f"💾 New best model saved with mAP@50: {best_map:.4f}")

    logging.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default="/sentinelkilndb_bechmarking_data/train")
    parser.add_argument('--val_path', type=str, default="/sentinelkilndb_bechmarking_data/val")
    parser.add_argument('--test_path', type=str, default="/sentinelkilndb_bechmarking_data/test")
    parser.add_argument('--output_dir', type=str, default="/work_dirs/copernicus_train")
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    args = parser.parse_args()
    main(args)