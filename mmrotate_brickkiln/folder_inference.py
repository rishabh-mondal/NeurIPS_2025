from mmrotate.apis import inference_detector_by_patches
from mmdet.apis import init_detector, inference_detector
import os
from tqdm import tqdm
import cv2
import numpy as np
import shutil
from mmcv.ops import nms_rotated
import torch


def init_model(config_file, checkpoint_file, device='cuda:0'):
    """Initialize and return the detector model."""
    model = init_detector(config_file, checkpoint_file, device=device)
    class_names = model.CLASSES
    print(f"Model loaded with classes: {class_names}")
    return model, class_names


def apply_rotated_nms(bboxes, scores, labels, nms_iou_threshold=0.3):
    if len(bboxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Convert lists to numpy arrays if needed
    bboxes = np.array(bboxes, dtype=np.float32)  # Shape (N, 8)
    scores = np.array(scores, dtype=np.float32)  # Shape (N,)
    labels = np.array(labels, dtype=np.int64)    # Shape (N,)

    # Convert to tensor for PyTorch operations
    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)  # (N, 8)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)  # (N,)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)    # (N,)

    _, keep_indices = nms_rotated(bboxes_tensor, scores_tensor, nms_iou_threshold)

    # Filtered results
    bboxes_filtered = bboxes_tensor[keep_indices].numpy()
    scores_filtered = scores_tensor[keep_indices].numpy()
    labels_filtered = labels_tensor[keep_indices].numpy()

    return bboxes_filtered, scores_filtered, labels_filtered

def convert_to_obb(center, w, h, angle):
    # Convert angle from radians to degrees
    angle_deg = angle * 180.0 / np.pi
    # Get the rotated bounding box points
    rect = ((center[0], center[1]), (w, h), angle_deg)
    points = cv2.boxPoints(rect)  # Points of the rotated box in (x1, y1, x2, y2, x3, y3, x4, y4) format
    return points.flatten()


def process_image(image_path, model, class_names, sizes, steps, ratios, merge_iou_thr,
                  score_thr, apply_nms=True, nms_iou_threshold=0.3):
    """Process a single image and return its DOTA-style bounding boxes."""
    # result = inference_detector_by_patches(model, image_path, sizes, steps, ratios, merge_iou_thr)    # Inference on patches 
    result = inference_detector(model, image_path)  # Inference on the whole image

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    dota_bboxes = []
    scores = []
    labels = []
    
    for class_id, bboxes in enumerate(result):  # For each class, bboxes are already inferred
        # Step 1: Convert each bounding box to 8 points in DOTA format
        for bbox in bboxes:
            if bbox[5] >= score_thr:  # Only consider boxes with score above threshold
                cx, cy, w, h, angle, score = bbox[:6]
                points = convert_to_obb((cx, cy), w, h, angle)  # Convert to DOTA format (8 points)
                dota_bboxes.append(points)
                scores.append(score)
                labels.append(class_id)

    if apply_nms:
        # Step 2: Apply rotated NMS on DOTA format boxes
        filtered_bboxes, filtered_scores, filtered_labels = apply_rotated_nms(
            dota_bboxes, scores, labels, nms_iou_threshold
        )
    else:
        filtered_bboxes, filtered_scores, filtered_labels = np.array(dota_bboxes), np.array(scores), np.array(labels)

    output_data = []
    # Step 3: Collect results after NMS in DOTA format (8 points)
    for i in range(len(filtered_bboxes)):
        bbox = filtered_bboxes[i]
        score = filtered_scores[i]
        label = filtered_labels[i]
        class_name = class_names[label]
        
        points_str = ' '.join([f"{p:.1f}" for p in bbox])  # Convert points to string
        output_data.append(f"{points_str} {class_name} {score:.4f}")

    return image_name, output_data


def save_results(output_dir, image_name, output_data):
    """Save the DOTA-style results to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    output_txt_path = os.path.join(output_dir, f'{image_name}.txt')
    with open(output_txt_path, 'w') as f:
        for line in output_data:
            f.write(line + '\n')


def process_folder(image_dir, output_dir, model, class_names, sizes, steps, ratios,
                   merge_iou_thr, score_thr, apply_nms=True, nms_iou_threshold=0.3):
    """Process all images in the image directory."""
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
    ])

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        image_name, output_data = process_image(
            image_path, model, class_names, sizes, steps, ratios,
            merge_iou_thr, score_thr, apply_nms, nms_iou_threshold
        )
        save_results(output_dir, image_name, output_data)


if __name__ == "__main__":
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

    # Configuration

    # config_file = 'configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_brickkiln_le90.py'
    # checkpoint_folder = 'work_dirs/oriented_rcnn_r50_fpn_1x_brickkiln_le90'
    # image_dir = '../data/sentinel/test/images'  # Directory containing images
    # modelname = 'test_oriented_rcnn'

    # config_file = 'configs/redet/redet_re50_refpn_1x_brickkiln_le90.py'
    # checkpoint_folder = 'work_dirs/redet_re50_refpn_1x_brickkiln_le90'
    # image_dir = '../data/stratified_split/test/images'  # Directory containing images
    # modelname = 'test_redet_re50_refpn'

    # config_file = 'configs/s2anet/s2anet_r50_fpn_1x_brickkiln_le135.py'
    # checkpoint_folder = 'work_dirs/s2anet_r50_fpn_1x_brickkiln_le135'
    # image_dir = '../data/stratified_split/test/images'  # Directory containing images
    # modelname = 'test_s2anet_r50'

    # Patch inference parameters
    sizes = [128]         # Patch size(s)
    steps = [64]          # Step size(s) for sliding window
    ratios = [1.0]        # Image resizing ratios
    merge_iou_thr = 0.1   # IoU threshold for merging results

    score_thr = 0.1      # Score threshold for saving results
    apply_nms = True
    nms_iou_threshold = 0.5

    # epoch = 1
    checkpoint_file = os.path.join(checkpoint_folder, f'epoch_{epoch}.pth')
    output_dir = f'results/{modelname}_nms_{nms_iou_threshold}_conf_{score_thr}/epoch_{epoch}/annfiles'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Config file: {config_file}")
    print(f"Checkpoint file: {checkpoint_file}")
    model, class_names = init_model(config_file, checkpoint_file)
    process_folder(image_dir, output_dir, model, class_names, sizes, steps, ratios,
                   merge_iou_thr, score_thr, apply_nms, nms_iou_threshold)

    print("All images processed.")
