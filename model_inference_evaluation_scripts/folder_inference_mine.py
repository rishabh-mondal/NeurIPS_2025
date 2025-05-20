from mmrotate.apis import inference_detector_by_patches
from mmdet.apis import init_detector, inference_detector
import os
from tqdm import tqdm
import cv2
import numpy as np
import shutil
from mmcv.ops import nms_rotated
import torch
from torch.cuda.amp import autocast


def init_model(config_file, checkpoint_file, device='cuda:0'):
    """Initialize and return the detector model."""
    model = init_detector(config_file, checkpoint_file, device=device)
    class_names = ['CFCBK', 'FCBK', 'Zigzag']  # Replace with model.CLASSES if needed
    print(f"Model loaded with classes: {class_names}")
    return model, class_names


def apply_rotated_nms(bboxes, scores, labels, nms_iou_threshold=0.1):
    if len(bboxes) == 0:
        return np.array([]), np.array([]), np.array([])

    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    _, keep_indices = nms_rotated(bboxes_tensor, scores_tensor, nms_iou_threshold)

    return (
        bboxes_tensor[keep_indices].numpy(),
        scores_tensor[keep_indices].numpy(),
        labels_tensor[keep_indices].numpy()
    )


def convert_to_obb(center, w, h, angle):
    angle_deg = angle * 180.0 / np.pi
    rect = ((center[0], center[1]), (w, h), angle_deg)
    points = cv2.boxPoints(rect)
    return points.flatten()


def process_image(image_path, model, class_names, sizes, steps, ratios, merge_iou_thr,
                  score_thr, apply_nms=True, nms_iou_threshold=0.50):
    """Process a single image and return its DOTA-style bounding boxes."""
    with autocast():
        result = inference_detector(model, image_path)

    pred_instances = result.pred_instances  # New API returns DetDataSample object
    bboxes_with_scores = pred_instances.bboxes.cpu().numpy()  # cx, cy, w, h, angle
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()

    dota_bboxes = []
    final_scores = []
    final_labels = []

    for i in range(len(bboxes_with_scores)):
        if scores[i] >= score_thr:
            cx, cy, w, h, angle = bboxes_with_scores[i]
            polygon = convert_to_obb((cx, cy), w, h, angle)
            dota_bboxes.append(polygon)
            final_scores.append(scores[i])
            final_labels.append(labels[i])

    if apply_nms:
        filtered_bboxes, filtered_scores, filtered_labels = apply_rotated_nms(
            dota_bboxes, final_scores, final_labels, nms_iou_threshold
        )
    else:
        filtered_bboxes = np.array(dota_bboxes)
        filtered_scores = np.array(final_scores)
        filtered_labels = np.array(final_labels)

    output_data = []
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(len(filtered_bboxes)):
        polygon = filtered_bboxes[i]
        score = filtered_scores[i]
        label = filtered_labels[i]
        class_name = class_names[label]
        points_str = ' '.join([f"{pt:.1f}" for pt in polygon])
        output_data.append(f"{points_str} {class_name} {score:.4f}")

    return image_name, output_data


def save_results(output_dir, image_name, output_data):
    os.makedirs(output_dir, exist_ok=True)
    output_txt_path = os.path.join(output_dir, f'{image_name}.txt')
    with open(output_txt_path, 'w') as f:
        for line in output_data:
            f.write(line + '\n')


def process_folder(image_dir, output_dir, model, class_names, sizes, steps, ratios,
                   merge_iou_thr, score_thr, apply_nms=True, nms_iou_threshold=0.50):
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
    # Configuration
    config_file = '../mmrotate_brickkiln/configs/dcfl/dcfl-le90_r50_1x_dota.py'
    checkpoint_folder = '../mmrotate_brickkiln/work_dirs_saved/dcfl_90'
    image_dir = '../mmrotate_brickkiln/data/dota/test/images'
    modelname = os.path.splitext(os.path.basename(config_file))[0]
    print(f"Model name: {modelname}")
    sizes = [128]
    steps = [64]
    ratios = [1.0]
    merge_iou_thr = 0.1
    score_thr = 0.1
    apply_nms = True
    nms_iou_threshold = 0.50
    epoch = 18
    checkpoint_file = os.path.join(checkpoint_folder, f'epoch_{epoch}.pth')
    output_dir = f'results/test_{modelname}_nms_{nms_iou_threshold}_conf_{score_thr}/epoch_{epoch}/annfiles'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    model, class_names = init_model(config_file, checkpoint_file)
    process_folder(image_dir, output_dir, model, class_names, sizes, steps, ratios,
                   merge_iou_thr, score_thr, apply_nms, nms_iou_threshold)

    print("All images processed.")
