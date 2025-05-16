from mmrotate.apis import inference_detector_by_patches
from mmdet.apis import init_detector, show_result_pyplot

# Specify the paths to your config and checkpoint files
config_file = 'configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_brickkiln_le90.py'
checkpoint_file = 'work_dirs/oriented_rcnn_r50_fpn_1x_brickkiln_le90_first/latest.pth'
image_path = '../data/sentinel/test/images/28.2090_77.4057.png'  # Replace 'your_image.jpg' with the actual image filename

# Initialize the detector
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or 'cpu'

# Example values for patch inference
sizes = [128]         # Patch size(s)
steps = [64]          # Step size(s) for sliding window
ratios = [1.0]        # Image resizing ratios
merge_iou_thr = 0.1   # IoU threshold for merging results

result = inference_detector_by_patches(model, image_path, sizes, steps, ratios, merge_iou_thr)
print(result)

# Visualize the results
show_result_pyplot(model, image_path, result, score_thr=0.3, out_file='output.jpg')
