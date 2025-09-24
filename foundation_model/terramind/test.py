import os
import sys
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

# --- Import your training modules ---
from train_terramind import BrickKilnDataset, create_model, collate_fn

# --- Setup Logging ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# --- Evaluate ---
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    loop = tqdm(data_loader, desc="Evaluating on Test Set")

    metric_class_wise = MeanAveragePrecision(box_format='xyxy', class_metrics=True, iou_thresholds=[0.5])
    metric_forced_agnostic = MeanAveragePrecision(box_format='xyxy', class_metrics=True, iou_thresholds=[0.5])

    class_names = {1: 'CFCBK', 2: 'FCBK', 3: 'Zigzag'}

    for images, targets in loop:
        images = [img.to(device) for img in images]
        predictions = model(images)

        preds_cpu = [{k: v.to("cpu") for k, v in p.items()} for p in predictions]
        tgts_cpu = [{k: v.to("cpu") for k, v in t.items()} for t in targets]

        # update metrics
        metric_class_wise.update(preds_cpu, tgts_cpu)

        agnostic_preds = [{'boxes': p['boxes'], 'scores': p['scores'], 'labels': torch.ones_like(p['labels'])} for p in preds_cpu]
        agnostic_targets = [{'boxes': t['boxes'], 'labels': torch.ones_like(t['labels'])} for t in tgts_cpu]
        metric_forced_agnostic.update(agnostic_preds, agnostic_targets)

    # --- Compute final metrics ---
    class_wise_results = metric_class_wise.compute()
    forced_agnostic_results = metric_forced_agnostic.compute()

    ca_map_50 = forced_agnostic_results['map_50'].item() * 100
    class_ids = class_wise_results['classes'].tolist()
    map_50_per_class_list = class_wise_results['map_per_class'].tolist()
    class_map50_dict = {cid: val * 100 for cid, val in zip(class_ids, map_50_per_class_list)}

    cfcbk_map50 = class_map50_dict.get(1, 0.0)
    fcbk_map50 = class_map50_dict.get(2, 0.0)
    zigzag_map50 = class_map50_dict.get(3, 0.0)

    # --- Print results directly ---
    print("\n" + "="*60)
    print("📊 SENTINELKILNDB Benchmark Results (Test Set)")
    print("="*60)
    print(f"{'CA mAP50':<15}{'CFCBK mAP50':<15}{'FCBK mAP50':<15}{'Zigzag mAP50':<15}")
    print("-"*60)
    print(f"{ca_map_50:<15.2f}{cfcbk_map50:<15.2f}{fcbk_map50:<15.2f}{zigzag_map50:<15.2f}")
    print("="*60 + "\n")


def run_test():
    setup_logging()

    weights = "/work_dirs/terramind_train/best_model.pth"
    data_root = "/sentinelkilndb_bechmarking_data/test"

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Dataset ---
    logging.info("Loading test dataset (OBB labels)...")
    test_dataset = BrickKilnDataset(root=data_root, split="test")  
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # --- Load Model ---
    logging.info("Creating TerraMind model structure...")
    model = create_model(num_classes=4).to(device)  # 3 classes + background

    logging.info(f"Loading weights from {weights}")
    model.load_state_dict(torch.load(weights, map_location=device))

    # --- Run Eval ---
    evaluate(model, test_loader, device)

# run in notebook cell:
run_test()
