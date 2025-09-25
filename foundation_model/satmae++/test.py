# test_satmaepp.py
import os
import sys
import logging
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import pandas as pd

from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

# --- Import your training components ---
# If your training file is named differently, change the import below.
from satmaepp_train import (
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

    # Metric 1: class-wise mAP@50 (CFCBK, FCBK, Zigzag)
    metric_class_wise = MeanAveragePrecision(
        box_format="xyxy", class_metrics=True, iou_thresholds=[0.5]
    )
    # Metric 2: forced class-agnostic mAP@50
    metric_forced_agnostic = MeanAveragePrecision(
        box_format="xyxy", class_metrics=True, iou_thresholds=[0.5]
    )

    rows = []
    class_names = {1: "CFCBK", 2: "FCBK", 3: "Zigzag"} 

    for batch in loop:
        img_names, images, targets = batch
        if images is None:
            continue

        images = [img.to(device, non_blocking=True) for img in images]
        preds = model(images)

        preds_cpu = [{k: v.to("cpu") for k, v in p.items()} for p in preds]
        tgts_cpu = [{k: v.to("cpu") for k, v in t.items()} for t in targets]

        # class-wise metric
        metric_class_wise.update(preds_cpu, tgts_cpu)

        # forced-agnostic metric (all labels -> 1)
        agn_preds = [
            {"boxes": p["boxes"], "scores": p["scores"], "labels": torch.ones_like(p["labels"])}
            for p in preds_cpu
        ]
        agn_tgts = [{"boxes": t["boxes"], "labels": torch.ones_like(t["labels"])} for t in tgts_cpu]
        metric_forced_agnostic.update(agn_preds, agn_tgts)

        # CSV rows
        for i in range(len(images)):
            name = img_names[i]
            gt = tgts_cpu[i]
            pr = preds_cpu[i]

            for j in range(len(gt["boxes"])):
                cid = int(gt["labels"][j].item())
                x1, y1, x2, y2 = gt["boxes"][j].tolist()
                rows.append(
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

            for j in range(len(pr["boxes"])):
                cid = int(pr["labels"][j].item())
                score = float(pr["scores"][j].item())
                x1, y1, x2, y2 = pr["boxes"][j].tolist()
                rows.append(
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

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "test_results_raw.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logging.info(f"Saved raw detection CSV to: {csv_path}")

    # compute metrics
    try:
        cls_res = metric_class_wise.compute()
        agn_res = metric_forced_agnostic.compute()

        ca_map50 = float(agn_res["map_50"].item() * 100.0)

        class_ids = cls_res.get("classes", torch.tensor([])).tolist()
        map_per_class = cls_res.get("map_per_class", torch.tensor([])).tolist()
        class_map50 = {int(cid): 100.0 * float(v) for cid, v in zip(class_ids, map_per_class)}

        cfcbk = class_map50.get(1, 0.0)
        fcbk = class_map50.get(2, 0.0)
        zigzag = class_map50.get(3, 0.0)

        logging.info("\n" + "=" * 60)
        logging.info("SENTINELKILNDB Benchmark (SatMAE++ detector)")
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    logging.info(f"Using device: {device}")

    logging.info("Loading test dataset...")
    test_ds = BrickKilnDatasetForTest(args.data_root, split="test", input_size=args.input_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_name,
        pin_memory=True,
    )

    logging.info("Building SatMAE++ detector...")
    model = create_model(
        num_classes=4,                
        image_size=args.input_size,   
        use_bgr_rgb_quirk=False
    ).to(device)

    logging.info(f"Loading fine-tuned weights: {weights_path}")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)

    out_dir = weights_path.parent
    evaluate(model, test_loader, device, out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a trained SatMAE++ Faster R-CNN on the test split.")
    p.add_argument("--weights", type=str, default="/work_dirs/satmaepp_train/best_model3.pth", help="Path to fine-tuned weights (e.g., best_model3.pth).")
    # IMPORTANT: point this to the **test** directory, e.g., .../sentinelkilndb_bechmarking_data/test
    p.add_argument("--data_root", type=str, default="/sentinelkilndb_bechmarking_data/test", help="Root of the test split (contains images/ and yolo_obb_labels/).")
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--input_size", type=int, default=224, help="Must match the training input_size.")
    args = p.parse_args()
    main(args)
