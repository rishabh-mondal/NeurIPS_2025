from folder_inference import init_model, process_folder
import os
import gc
import torch

# conda activate open-mmlab
# export CUDA_VISIBLE_DEVICES=1
# set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# # Configuration

# config_file = 'configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_brickkiln_le90.py'
# checkpoint_folder = 'work_dirs/oriented_rcnn_r50_fpn_1x_brickkiln_le90'
# image_dir = '../data/stratified_split/test/images'
# modelname = 'test_oriented_rcnn'
# # nohup python folder_inference_multi_epoch.py > oriented_rcnn_inference.log 2>&1 &

# config_file = 'configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_brickkiln_le90.py'
# checkpoint_folder = 'work_dirs/rotated_retinanet_obb_r50_fpn_1x_brickkiln_le90'
# image_dir = '../data/stratified_split/test/images'
# modelname = 'test_rotated_retinanet_obb'
# # nohup python folder_inference_multi_epoch.py > rotated_retinanet_obb_inference.log 2>&1 &

# config_file = 'configs/redet/redet_re50_refpn_1x_brickkiln_le90.py'
# checkpoint_folder = 'work_dirs/redet_re50_refpn_1x_brickkiln_le90'
# image_dir = '../data/stratified_split/test/images'
# modelname = 'test_redet_re50_refpn'
# # nohup python folder_inference_multi_epoch.py > redet_re50_refpn_inference.log 2>&1 &

# config_file = 'configs/roi_trans/roi_trans_swin_tiny_fpn_1x_brickkiln_le90.py'
# checkpoint_folder = 'work_dirs/roi_trans_swin_tiny_fpn_1x_brickkiln_le90'
# image_dir = '../data/stratified_split/test/images'
# modelname = 'test_roi_trans_swin_tiny_fpn'
# # nohup python folder_inference_multi_epoch.py > roi_trans_swin_tiny_fpn_inference.log 2>&1 &

# config_file = 'configs/s2anet/s2anet_r50_fpn_1x_brickkiln_le135.py'
# checkpoint_folder = 'work_dirs/s2anet_r50_fpn_1x_brickkiln_le135'
# image_dir = '../data/stratified_split/test/images'
# modelname = 'test_s2anet_r50'
# # nohup python folder_inference_multi_epoch.py > s2anet_r50_fpn_1x_brickkiln_le135_inference.log 2>&1 &

# config_file = 'work_dirs/gliding_vertex_r50_fpn_1x_brickkiln_le90/gliding_vertex_r50_fpn_1x_brickkiln_le90.py'
# checkpoint_folder = 'work_dirs/gliding_vertex_r50_fpn_1x_brickkiln_le90'
# image_dir = '../data/stratified_split/test/images'
# modelname = 'test_gliding_vertex_r50_fpn_1x_brickkiln_le90'
# # nohup python folder_inference_multi_epoch.py > gliding_vertex_r50_fpn_1x_brickkiln_le90_inference.log 2>&1 &

# config_file = 'work_dirs/r3det_r50_fpn_1x_brickkiln_oc/r3det_r50_fpn_1x_brickkiln_oc.py'
# checkpoint_folder = 'work_dirs/r3det_r50_fpn_1x_brickkiln_oc'
# image_dir = '../data/stratified_split/test/images'
# modelname = 'test_r3det_r50_fpn_1x_brickkiln_oc'
# # nohup python folder_inference_multi_epoch.py > r3det_r50_fpn_1x_brickkiln_oc_inference.log 2>&1 &

# config_file = 'work_dirs/r3det_tiny_r50_fpn_1x_brickkiln_oc/r3det_tiny_r50_fpn_1x_brickkiln_oc.py'
# checkpoint_folder = 'work_dirs/r3det_tiny_r50_fpn_1x_brickkiln_oc'
# image_dir = '../data/stratified_split/test/images'
# modelname = 'test_r3det_tiny_r50_fpn_1x_brickkiln_oc'
# # nohup python folder_inference_multi_epoch.py > r3det_tiny_r50_fpn_1x_brickkiln_oc_inference.log 2>&1 &

# config_file = 'work_dirs/roi_trans_swin_tiny_fpn_1x_up_le90/roi_trans_swin_tiny_fpn_1x_up_le90.py'
# checkpoint_folder = 'work_dirs/roi_trans_swin_tiny_fpn_1x_up_le90'
# image_dir = '../data/uttar_pradesh/test/images'
# modelname = 'test_roi_trans_swin_tiny_fpn_1x_up_le90'
# # nohup python folder_inference_multi_epoch.py > roi_trans_swin_tiny_fpn_1x_up_le90_inference.log 2>&1 &

# config_file = 'work_dirs/roi_trans_swin_tiny_fpn_1x_up_le90/roi_trans_swin_tiny_fpn_1x_up_le90.py'
# checkpoint_folder = 'work_dirs/roi_trans_swin_tiny_fpn_1x_up_le90'
# image_dir = '../data/dhaka/images'
# modelname = 'test_roi_trans_swin_tiny_fpn_1x_up_on_dhaka'
# # nohup python folder_inference_multi_epoch.py > roi_trans_swin_tiny_fpn_1x_up_on_dhaka_inference.log 2>&1 &

# config_file = 'work_dirs/roi_trans_swin_tiny_fpn_1x_up_le90/roi_trans_swin_tiny_fpn_1x_up_le90.py'
# checkpoint_folder = 'work_dirs/roi_trans_swin_tiny_fpn_1x_up_le90'
# image_dir = '../data/pak_punjab/images'
# modelname = 'test_roi_trans_swin_tiny_fpn_1x_up_on_pakpunjab'
# # nohup python folder_inference_multi_epoch.py > roi_trans_swin_tiny_fpn_1x_up_on_pakpunjab_inference.log 2>&1 &


# Patch inference parameters
sizes = [128]         # Patch size(s)
steps = [64]          # Step size(s) for sliding window
ratios = [1.0]        # Image resizing ratios
merge_iou_thr = 0.1   # IoU threshold for merging results


score_thr = 0.1      # Score threshold for saving results
nms_iou_thr = 0.5
# Set this to True to apply NMS, False to skip NMS
APPLY_NMS = True
# Epochs to process (12 to 1)
max_epochs = 25
min_epochs = 25

epochs = list(range(min_epochs, max_epochs + 1))[::-1]  # Reverse order

# print stuff
print(f"Config file: {config_file}")
print(f"Checkpoint folder: {checkpoint_folder}")
print(f"Image directory: {image_dir}")
print(f"Model name: {modelname}")
print(f"Patch sizes: {sizes}")
print(f"Step sizes: {steps}")
print(f"Ratios: {ratios}")
print(f"Merging IoU threshold: {merge_iou_thr}")
print(f"Score threshold: {score_thr}")
print(f"NMS IoU threshold: {nms_iou_thr}")
print(f"Apply NMS: {APPLY_NMS}")

for epoch in epochs:
    print(f"Processing epoch {epoch}...")
    checkpoint_file = os.path.join(checkpoint_folder, f'epoch_{epoch}.pth')
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint {checkpoint_file} does not exist, skipping.")
        continue
    if APPLY_NMS:
        output_dir = f'results/{modelname}_nms_{nms_iou_thr}_conf_{score_thr}/epoch_{epoch}/annfiles'  # Directory to save DOTA-style labels
    else:
        output_dir = f'results/{modelname}_no_nms_conf_{score_thr}/epoch_{epoch}/annfiles'  # Directory to save DOTA-style labels
    print(f"\nProcessing epoch {epoch}...")
    model, class_names = init_model(config_file, checkpoint_file, device=device)
    process_folder(image_dir, output_dir, model, class_names, sizes, steps, ratios, merge_iou_thr, score_thr, apply_nms=APPLY_NMS, nms_iou_threshold=nms_iou_thr)
    print(f"Epoch {epoch} processed. Results saved to {output_dir}.")
    # delete all variables
    del model, class_names, checkpoint_file
    # clear cache
    gc.collect()
    torch.cuda.empty_cache()

print("All epochs processed.")
