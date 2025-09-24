import os
import sys
import logging
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import pandas as pd

# --- TorchVision & TorchMetrics Imports ---
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

# --- Import our custom modules from the Galileo training script ---
from train_galelio import BrickKilnDataset as OriginalBrickKilnDataset
from train_galelio import create_model, collate_fn as original_collate_fn

# --- Modified Dataset to include image names for CSV logging ---
class BrickKilnDatasetForTest(OriginalBrickKilnDataset):
    def __getitem__(self, idx: int):
        image_tensor, target = super().__getitem__(idx)
        img_name = self.img_files[idx]
        return img_name, image_tensor, target

# --- Modified Collate Function to handle image names ---
def collate_fn_with_name(batch):
    batch = [item for item in batch if item[2]['boxes'].shape[0] > 0]
    if not batch: return None, None, None
    
    img_names = [item[0] for item in batch]
    images_and_targets = [(item[1], item[2]) for item in batch]
    images, targets = original_collate_fn(images_and_targets)
    
    return img_names, images, targets

# --- Setup Logging ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# --- The Final, Rigorously Corrected 'evaluate' Function ---
@torch.no_grad()
def evaluate(model, data_loader, device, output_dir):
    model.eval()
    loop = tqdm(data_loader, desc="Evaluating on Test Set")
    
    metric_class_wise = MeanAveragePrecision(box_format='xyxy', class_metrics=True, iou_thresholds=[0.5])
    metric_forced_agnostic = MeanAveragePrecision(box_format='xyxy', class_metrics=True, iou_thresholds=[0.5])

    results_list = []
    class_names = {1: 'CFCBK', 2: 'FCBK', 3: 'Zigzag'}

    for batch in loop:
        img_names, images, targets = batch
        if images is None: continue
        
        images = [img.to(device) for img in images]
        predictions = model(images)
        
        predictions_cpu = [{k: v.to('cpu') for k, v in p.items()} for p in predictions]
        targets_cpu = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
        
        metric_class_wise.update(predictions_cpu, targets_cpu)
        
        agnostic_preds = [{'boxes': p['boxes'], 'scores': p['scores'], 'labels': torch.ones_like(p['labels'])} for p in predictions_cpu]
        agnostic_targets = [{'boxes': t['boxes'], 'labels': torch.ones_like(t['labels'])} for t in targets_cpu]
        metric_forced_agnostic.update(agnostic_preds, agnostic_targets)

        # CSV Logging Logic...
        for i in range(len(images)):
            img_name, gt_targets, pred_output = img_names[i], targets_cpu[i], predictions_cpu[i]
            gt_boxes, gt_labels = gt_targets['boxes'], gt_targets['labels']
            pred_boxes, pred_labels, pred_scores = pred_output['boxes'], pred_output['labels'], pred_output['scores']
            for j in range(len(gt_boxes)):
                class_id = gt_labels[j].item()
                results_list.append({'image_name': img_name, 'box_type': 'ground_truth', 'class_id': class_id, 'class_name': class_names.get(class_id, 'N/A'), 'confidence_score': 1.0, 'bbox_xmin': gt_boxes[j][0].item(), 'bbox_ymin': gt_boxes[j][1].item(), 'bbox_xmax': gt_boxes[j][2].item(), 'bbox_ymax': gt_boxes[j][3].item()})
            for j in range(len(pred_boxes)):
                class_id = pred_labels[j].item()
                results_list.append({'image_name': img_name, 'box_type': 'prediction', 'class_id': class_id, 'class_name': class_names.get(class_id, 'N/A'), 'confidence_score': pred_scores[j].item(), 'bbox_xmin': pred_boxes[j][0].item(), 'bbox_ymin': pred_boxes[j][1].item(), 'bbox_xmax': pred_boxes[j][2].item(), 'bbox_ymax': pred_boxes[j][3].item()})

    output_csv_path = output_dir / "test_results_final.csv"
    pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
    logging.info(f"Successfully saved raw detection results to {output_csv_path}")

    # --- Compute and Display Final, Paper-Ready Metrics ---
    try:
        class_wise_results = metric_class_wise.compute()
        forced_agnostic_results = metric_forced_agnostic.compute()

        ca_map_50 = forced_agnostic_results['map_50'].item() * 100
        
        class_ids = class_wise_results['classes'].tolist()
        map_50_per_class_list = class_wise_results['map_per_class'].tolist()
        class_map50_dict = {cid: val * 100 for cid, val in zip(class_ids, map_50_per_class_list)}

        cfcbk_map50 = class_map50_dict.get(1, 0.0)
        fcbk_map50 = class_map50_dict.get(2, 0.0)
        zigzag_map50 = class_map50_dict.get(3, 0.0)

        logging.info("\n\n" + "="*60)
        logging.info("      SENTINELKILNDB Benchmark Results (Forced Agnostic)")
        logging.info("="*60)
        
        header = f"{'CA mAP50':<15}{'CFCBK mAP50':<15}{'FCBK mAP50':<15}{'Zigzag mAP50':<15}"
        values = f"{ca_map_50:<15.2f}{cfcbk_map50:<15.2f}{fcbk_map50:<15.2f}{zigzag_map50:<15.2f}"
        
        logging.info(f"\n{header}\n{'-'*60}\n{values}\n")
        logging.info("="*60 + "\n")
        
    except Exception as e:
        logging.error(f"Could not compute final test metrics: {e}", exc_info=True)


def main(args):
    setup_logging()
    
    if not os.path.exists(args.weights):
        logging.error(f"Weights file not found at: {args.weights}")
        sys.exit(1)
        
    output_dir = Path(args.weights).parent
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    logging.info("Loading test dataset...")
    test_dataset = BrickKilnDatasetForTest(args.data_root, 'test', input_size=args.input_size) 
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=collate_fn_with_name, 
        pin_memory=True
    )

    logging.info("Creating Galileo model structure...")
    # This is the key change. We now pass the required 'weights_path' to create the model structure.
    model = create_model(weights_path=args.backbone_init_path, num_classes=4).to(device)

    logging.info(f"Loading final fine-tuned weights from: {args.weights}")
    # This load_state_dict call correctly OVERWRITES the initial backbone weights with your final ones.
    model.load_state_dict(torch.load(args.weights, map_location=device))

    evaluate(model, test_loader, device, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Galileo detector and generate paper-ready benchmarks.")
    # The final weights from your training run. This is the most important argument.
    parser.add_argument('--weights', type=str,default="/work_dirs/galelio_training/best_model.pth", help='Path to the final fine-tuned model weights (e.g., best_model.pth).')
    # The initial weights needed only to build the model's structure due to the function's requirement.
    parser.add_argument('--backbone_init_path', type=str, default="pre_trained_weights/nano", help='Path to original backbone weights needed to initialize the model structure (e.g., ./weights/nano).')
    parser.add_argument('--data_root', type=str, default="/sentinelkilndb_bechmarking_data/test", help='Root directory of the dataset.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation (e.g., cuda:0).')
    parser.add_argument('--input_size', type=int, default=128, help='Image size the model was trained on.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers.')
    
    args = parser.parse_args()
    main(args)