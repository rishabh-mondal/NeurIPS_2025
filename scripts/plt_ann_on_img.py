# plot the annotation file and image path and plot the boxes on img
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import shutil

METAINFO = {
    'classes': ('CFCBK', 'FCBK', 'Zigzag'),
    'palette': [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
}

def draw_annotations(image, ann_file, img_size, is_ann_normalized, ann_format='dota', is_label_number=False, is_prediction=False, add_score=False,
                   score_threshold=0.2, add_label=True):
    """Draws rotated bounding boxes from a DOTA or Supervision annotation file using specified colors."""
    img = image.copy()
    if not os.path.exists(ann_file):
        print(f"Annotation file {ann_file} not found.")
        return img

    class_to_color = dict(zip(METAINFO['classes'], METAINFO['palette']))
    
    with open(ann_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        values = line.strip().split()

        # Validate length based on format
        if ann_format == 'dota':
            if len(values) < 9:
                print(f"Invalid annotation line: {line} in {ann_file}")
                continue
            points = np.array(list(map(float, values[:8]))).reshape((4, 2))
            label = values[8]
            score = values[9] if is_prediction and len(values) > 9 else None

        elif ann_format == 'supervision':
            if len(values) < 9:
                print(f"Invalid annotation line: {line} in {ann_file}")
                continue
            label = values[0]
            points = np.array(list(map(float, values[1:9]))).reshape((4, 2))
            score = values[9] if is_prediction and len(values) > 9 else None

        else:
            print(f"Unsupported annotation format: {ann_format}")
            continue

        if is_label_number:
            label = METAINFO['classes'][int(label)]

        if is_ann_normalized:
            points *= img_size

        points = points.astype(int)
        color = class_to_color.get(label, (255, 255, 255))

        # Adjust label font scale and thickness based on image size
        font_scale = max(0.4, min(2.0, img.shape[0] / 512.0))
        thickness = max(1, int(img.shape[0] / 512))

        label_text = label
        if add_score and score is not None:
            label_text += f': {score}'

        if not is_prediction or (score is not None and float(score) > score_threshold):
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=1)
            if add_label:
                cv2.putText(img, label_text, (points[0][0], points[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        else:
            print(f"Skipping annotation with low score: {score} for {label}")

    return img



def annotate_directory(image_dir, ann_dir, out_dir, img_size, is_ann_normalized,
                       ann_format='dota', is_label_number=False, save=True, plot=False,
                       is_prediction=False, add_score=False, score_threshold=0.2, add_label=True):
    """Draws rotated bounding boxes from annotation files on images in a directory."""
    # remove the content of out_dir if it exists
    if os.path.exists(out_dir):
        for filename in os.listdir(out_dir):
            file_path = os.path.join(out_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    os.makedirs(out_dir, exist_ok=True)        

    image_files = sorted(os.listdir(image_dir))
    num_images = len(image_files)

    for i, image_name in enumerate(image_files):
        print(f'Processing image {i + 1}/{num_images}...')
        image_path = os.path.join(image_dir, image_name)

        if image_name.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
            image = np.array(Image.open(image_path).convert('RGB'))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = cv2.imread(image_path)

        if image is None:
            print(f"Image {image_path} not found.")
            continue

        ann_file = os.path.join(ann_dir, f'{os.path.splitext(image_name)[0]}.txt')
        annotated_img = draw_annotations(image, ann_file, img_size, is_ann_normalized,
                                         ann_format, is_label_number, is_prediction, add_score,
                                         score_threshold, add_label=add_label)

        if save:
            out_path = os.path.join(out_dir, image_name)
            success = cv2.imwrite(out_path, annotated_img)
            if not success:
                print(f"Failed to save annotated image to {out_path}")

        if plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


image_dir = '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/data/stratified_split/test/images'
ann_dir = '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/data/stratified_split/test/annfiles'
out_dir = '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/data/stratified_split/test/annotated_images'
img_size = 128
score_threshold = 0.1
is_prediction = False
add_score = False
add_label = False

# input_format = input("Enter the input format (yolo/dota/supervision): ").strip().lower()
input_format = 'dota'  # for above example
if input_format == 'yolo' or input_format == 'supervision':
    is_ann_normalized = True
    ann_format = 'supervision'
    is_label_number = True
elif input_format == 'dota':
    is_ann_normalized = False
    ann_format = 'dota'
    is_label_number = False


annotate_directory(image_dir, ann_dir, out_dir, img_size, is_ann_normalized, ann_format, is_label_number, save=True, plot=False, is_prediction=is_prediction, add_score=add_score, score_threshold=score_threshold, add_label=add_label)
