import os
import numpy as np
import supervision as sv

# Paths
GT_PATH = "gt/test_bihar_same_class_count_10_120_1000/labels"
IMAGE_PATH = "gt/test_bihar_same_class_count_10_120_1000/images"
PREDICTIONS_PATH = "results/train_bihar_test_bihar/annfiles"

def get_image_names_from_directory(directory):
    """Extracts image names (without extension) from a directory."""
    return {file_name.replace(".txt", "") for file_name in os.listdir(directory) if file_name.endswith(".txt")}

def load_detections(annotations_path, img_names, is_gt=True):
    """Loads detections only for images that exist in both GT and Predictions."""
    sv_data = []

    for image_id in img_names:
        file_path = os.path.join(annotations_path, f"{image_id}.txt")
        if not os.path.exists(file_path):  # Ensure file exists before processing
            continue

        xyxy_list = []
        class_ids = []
        scores = []

        with open(file_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            data = list(map(float, line.split()))
            class_id = int(data[0])
            polygon = np.array(data[1:9]).reshape(4, 2)  # Convert to (4,2) shape
            score = data[9] if not is_gt else 1.0  # Assign default confidence for GT

            # Convert quadrilateral to bounding box (min x, min y, max x, max y)
            x_min, y_min = np.min(polygon, axis=0)
            x_max, y_max = np.max(polygon, axis=0)
            bbox = [x_min, y_min, x_max, y_max]

            # Append to lists
            xyxy_list.append(bbox)
            class_ids.append(class_id)
            scores.append(score)

        # Convert lists into a Supervision Detections object
        detections = sv.Detections(
            xyxy=np.array(xyxy_list),
            class_id=np.array(class_ids),
            confidence=np.array(scores),
        )

        sv_data.append(detections)

    return sv_data


# Get common image names
gt_images = get_image_names_from_directory(GT_PATH)
pred_images = get_image_names_from_directory(PREDICTIONS_PATH)
common_images = gt_images.intersection(pred_images)

print(f"Total common images: {len(common_images)}")

# Load ground truth
targets = load_detections(GT_PATH, common_images, is_gt=True)
print(f"Loaded {len(targets)} ground truth detections.")

# Load predictions
predictions = load_detections(PREDICTIONS_PATH, common_images, is_gt=False)
print(f"Loaded {len(predictions)} prediction detections.")
