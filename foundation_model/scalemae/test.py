# test_scalemae.py
import os
import sys
import logging
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import pandas as pd

# TorchMetrics / DataLoader
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

# --- Import your training components (dataset, model, collate) ---
# Make sure train_scalemae_fixed.py is importable (same folder or on PYTHONPATH)
from scale_mae_temp import (
    BrickKilnDataset as OriginalBrickKilnDataset,
    create_model,
    collate_fn as original_collate_fn,
)

# ---------- Dataset wrapper to also return image names ----------
class BrickKilnDatasetForTest(OriginalBrickKilnDataset):
    def __init__(self, root: str, split: str, input_size: int = 224):
        super().__init__(root, split, input_size)
    def __getitem__(self, idx: int):
        image_tensor, target = super().__getitem__(idx)
        img_name = self.img_files[idx]
        return img_name, image_tensor, target

def collate_fn_with_name(batch):
    # keep only samples with at least one GT box
    batch = [b for b in batch if b[2]["boxes"].shape[0] > 0]
    if not batch:
        return None, None, None
    img_names = [b[0] for b in batch]
    images_targets = [(b[1], b[2]) for b in batch]
    images, targets = original_collate_fn(images_targets)
    return img_names, images, targets

# ---------- Logging ----------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

# ---------- Evaluation ----------
@torch.no_grad()
def evaluate(model, data_loader, device, output_dir: Path):
    model.eval()
    loop = tqdm(data_loader, desc="Evaluating on Test Set")

    # Metric 1: class-wise mAP@50 for CFCBK/FCBK/Zigzag
    metric_class_wise = MeanAveragePrecision(
        box_format="xyxy", class_metrics=True, iou_thresholds=[0.5]
    )
    # Metric 2: forced class-agnostic mAP@50
    metric_forced_agnostic = MeanAveragePrecision(
        box_format="xyxy", class_metrics=True, iou_thresholds=[0.5]
    )

    results_rows = []
    class_names = {1: "CFCBK", 2: "FCBK", 3: "Zigzag"}  # your label mapping

    for batch in loop:
        img_names, images, targets = batch
        if images is None:
            continue

        images = [img.to(device, non_blocking=True) for img in images]
        preds = model(images)

        preds_cpu = [{k: v.to("cpu") for k, v in p.items()} for p in preds]
        t_cpu = [{k: v.to("cpu") for k, v in t.items()} for t in targets]

        # Update class-wise metric
        metric_class_wise.update(preds_cpu, t_cpu)

        # Build forced-agnostic copies (all labels -> 1)
        agn_preds = [
            {"boxes": p["boxes"], "scores": p["scores"], "labels": torch.ones_like(p["labels"])}
            for p in preds_cpu
        ]
        agn_tgts = [{"boxes": t["boxes"], "labels": torch.ones_like(t["labels"])} for t in t_cpu]
        metric_forced_agnostic.update(agn_preds, agn_tgts)

        # Collect per-image GT and predictions for CSV
        for i in range(len(images)):
            name = img_names[i]
            gt = t_cpu[i]
            pr = preds_cpu[i]
            # GT rows
            for j in range(len(gt["boxes"])):
                cid = int(gt["labels"][j].item())
                x1, y1, x2, y2 = gt["boxes"][j].tolist()
                results_rows.append(
                    {
                        "image_name": name,
                        "box_type": "ground_truth",
                        "class_id": cid,
                        "class_name": class_names.get(cid, "N/A"),
                        "confidence_score": 1.0,
                        "bbox_xmin": x1,
                        "bbox_ymin": y1,
                        "bbox_xmax": x2,
                        "bbox_ymax": y2,
                    }
                )
            # Prediction rows
            for j in range(len(pr["boxes"])):
                cid = int(pr["labels"][j].item())
                score = float(pr["scores"][j].item())
                x1, y1, x2, y2 = pr["boxes"][j].tolist()
                results_rows.append(
                    {
                        "image_name": name,
                        "box_type": "prediction",
                        "class_id": cid,
                        "class_name": class_names.get(cid, "N/A"),
                        "confidence_score": score,
                        "bbox_xmin": x1,
                        "bbox_ymin": y1,
                        "bbox_xmax": x2,
                        "bbox_ymax": y2,
                    }
                )

    # Save raw detections
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "test_results_raw2.csv"
    pd.DataFrame(results_rows).to_csv(csv_path, index=False)
    logging.info(f"Saved raw detection CSV to: {csv_path}")

    # Compute metrics
    try:
        cls_res = metric_class_wise.compute()
        agn_res = metric_forced_agnostic.compute()

        ca_map50 = float(agn_res["map_50"].item() * 100.0)

        # class-wise mAP@50
        class_ids = cls_res["classes"].tolist() if "classes" in cls_res else []
        map_per_class = cls_res["map_per_class"].tolist() if "map_per_class" in cls_res else []
        class_map50 = {int(cid): 100.0 * float(v) for cid, v in zip(class_ids, map_per_class)}

        cfcbk = class_map50.get(1, 0.0)
        fcbk = class_map50.get(2, 0.0)
        zigzag = class_map50.get(3, 0.0)

        logging.info("\n" + "=" * 60)
        logging.info("SENTINELKILNDB Benchmark (Scale-MAE detector)")
        logging.info("=" * 60)
        header = f"{'CA mAP50':<15}{'CFCBK mAP50':<15}{'FCBK mAP50':<15}{'Zigzag mAP50':<15}"
        values = f"{ca_map50:<15.2f}{cfcbk:<15.2f}{fcbk:<15.2f}{zigzag:<15.2f}"
        logging.info(f"\n{header}\n{'-'*60}\n{values}\n")
        logging.info("=" * 60 + "\n")
    except Exception as e:
        logging.error(f"Metric computation failed: {e}", exc_info=True)

def main(args):
    setup_logging()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logging.error(f"Weights not found: {weights_path}")
        sys.exit(1)

    # Match the exact input size used in training (Scale-MAE wrapper asserts it)
    input_size = args.input_size

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    logging.info(f"Using device: {device}")

    # Dataset
    logging.info("Loading test dataset...")
    test_ds = BrickKilnDatasetForTest(args.data_root, "test", input_size=input_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_name,
        pin_memory=True,
    )

    # Model (must use same image_size as training)
    logging.info("Building Scale-MAE detector...")
    model = create_model(num_classes=4, image_size=input_size).to(device)

    logging.info(f"Loading fine-tuned weights: {weights_path}")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)

    out_dir = weights_path.parent
    evaluate(model, test_loader, device, out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a trained Scale-MAE Faster R-CNN on the test split.")
    p.add_argument("--weights", type=str, default='/work_dirs/scalemae_train/best_model2.pth', help="Path to best_model.pth from training.")
    p.add_argument("--data_root", type=str, default="/sentinelkilndb_bechmarking_data/test", help="Root containing test/images and test/yolo_obb_labels.")
    p.add_argument("--device", type=str, default="cuda:0", help="Device string, e.g., cuda:1")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--input_size", type=int, default=224, help="Must match training input_size (e.g., 224).")
    args = p.parse_args()
    main(args)
