import os
import sys
import logging
from pathlib import Path

import torch
from tqdm import tqdm

# --- TorchVision & TorchMetrics Imports ---
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

# --- Import our custom modules from the SatMAE training script ---
from train_satmae import BrickKilnDataset as OriginalBrickKilnDataset
from train_satmae import create_model, collate_fn as original_collate_fn


# --- Modified Dataset to include image names ---
class BrickKilnDatasetForTest(OriginalBrickKilnDataset):
    def __getitem__(self, idx: int):
        image_tensor, target = super().__getitem__(idx)
        img_name = self.img_files[idx]
        return img_name, image_tensor, target


# --- Modified Collate Function to handle image names ---
def collate_fn_with_name(batch):
    batch = [item for item in batch if item[2]['boxes'].shape[0] > 0]
    if not batch:
        return None, None, None

    img_names = [item[0] for item in batch]
    images_and_targets = [(item[1], item[2]) for item in batch]
    images, targets = original_collate_fn(images_and_targets)

    return img_names, images, targets


# --- Simple logging setup ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# --- Evaluate function ---
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    loop = tqdm(data_loader, desc="Evaluating on Test Set")

    # Metric #1: Class-wise mAP50
    metric_class_wise = MeanAveragePrecision(
        box_format='xyxy', class_metrics=True, iou_thresholds=[0.5]
    )

    # Metric #2: Class-agnostic mAP50 (forced single class)
    metric_forced_agnostic = MeanAveragePrecision(
        box_format='xyxy', class_metrics=True, iou_thresholds=[0.5]
    )

    class_names = {1: 'CFCBK', 2: 'FCBK', 3: 'Zigzag'}

    for batch in loop:
        img_names, images, targets = batch
        if images is None:
            continue

        images = [img.to(device) for img in images]
        predictions = model(images)

        predictions_cpu = [{k: v.to('cpu') for k, v in p.items()} for p in predictions]
        targets_cpu = [{k: v.to('cpu') for k, v in t.items()} for t in targets]

        # 1. Class-wise
        metric_class_wise.update(predictions_cpu, targets_cpu)

        # 2. Forced agnostic
        agnostic_preds = [
            {'boxes': p['boxes'], 'scores': p['scores'], 'labels': torch.ones_like(p['labels'])}
            for p in predictions_cpu
        ]
        agnostic_targets = [
            {'boxes': t['boxes'], 'labels': torch.ones_like(t['labels'])}
            for t in targets_cpu
        ]
        metric_forced_agnostic.update(agnostic_preds, agnostic_targets)

    # --- Compute metrics ---
    class_wise_results = metric_class_wise.compute()
    forced_agnostic_results = metric_forced_agnostic.compute()

    ca_map_50 = forced_agnostic_results['map_50'].item() * 100

    class_ids = class_wise_results['classes'].tolist()
    map_50_per_class_list = class_wise_results['map_per_class'].tolist()
    class_map50_dict = {cid: val * 100 for cid, val in zip(class_ids, map_50_per_class_list)}

    cfcbk_map50 = class_map50_dict.get(1, 0.0)
    fcbk_map50 = class_map50_dict.get(2, 0.0)
    zigzag_map50 = class_map50_dict.get(3, 0.0)

    # --- Nicely display results in notebook ---
    print("\n" + "="*60)
    print("      SENTINELKILNDB Benchmark Results (SatMAE)")
    print("="*60)
    print(f"{'CA mAP50':<15}{'CFCBK mAP50':<15}{'FCBK mAP50':<15}{'Zigzag mAP50':<15}")
    print("-"*60)
    print(f"{ca_map_50:<15.2f}{cfcbk_map50:<15.2f}{fcbk_map50:<15.2f}{zigzag_map50:<15.2f}")
    print("="*60 + "\n")


# --- Run in notebook ---
def run_satmae_eval():
    setup_logging()

    weights_path = "/work_dirs/best_model.pth"
    pretrained_backbone = "pre_trained_weights/satmae/fmow_pretrain.pth"
    data_root = "/sentinelkilndb_bechmarking_data/test"

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info("Loading test dataset...")
    test_dataset = BrickKilnDatasetForTest(data_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_with_name,
        pin_memory=True
    )

    logging.info("Creating SatMAE model...")
    model = create_model(weights_path=pretrained_backbone, num_classes=4).to(device)

    logging.info(f"Loading fine-tuned weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))

    evaluate(model, test_loader, device)


# In notebook, just call:
run_satmae_eval()
