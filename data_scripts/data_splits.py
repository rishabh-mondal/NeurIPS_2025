
import os
import shutil
import json
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

# ==== Base paths ====
base_path = "../data"
images_path = os.path.join(base_path, "images")
labels_path = os.path.join(base_path, "labels")
aa_labels_path = os.path.join(base_path, "aa_labels")
dota_labels_path = os.path.join(base_path, "dota_labels")

# ==== Output split directories ====
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(base_path, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_path, split, "labels"), exist_ok=True)
    os.makedirs(os.path.join(base_path, split, "aa_labels"), exist_ok=True)
    os.makedirs(os.path.join(base_path, split, "dota_labels"), exist_ok=True)

# ==== Step 1: Collect image file names and dominant class ====
image_files = []
class_ids = []

for label_file in os.listdir(labels_path):
    if not label_file.endswith('.txt'):
        continue

    label_fp = os.path.join(labels_path, label_file)
    try:
        with open(label_fp, 'r') as f:
            lines = f.readlines()
            if not lines:
                continue
            labels = [int(line.strip().split()[0]) for line in lines]
            most_common_class = Counter(labels).most_common(1)[0][0]
            image_name = os.path.splitext(label_file)[0] + ".png"
            if os.path.exists(os.path.join(images_path, image_name)):
                image_files.append(os.path.splitext(label_file)[0])
                class_ids.append(most_common_class)
    except Exception as e:
        print(f"⚠️ Error reading {label_file}: {e}")

# ==== Step 1.5: Filter classes with fewer than 2 samples ====
class_count = Counter(class_ids)
filtered_image_files = []
filtered_class_ids = []

for img, cls in zip(image_files, class_ids):
    if class_count[cls] > 1:
        filtered_image_files.append(img)
        filtered_class_ids.append(cls)

print(f"⚠️ Removed {len(image_files) - len(filtered_image_files)} images with rare classes")

image_files = filtered_image_files
class_ids = filtered_class_ids

# ==== Step 2: Stratified split (60:20:20) ====
splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
train_idx, temp_idx = next(splitter1.split(image_files, class_ids))

train_files = [image_files[i] for i in train_idx]
train_classes = [class_ids[i] for i in train_idx]

temp_files = [image_files[i] for i in temp_idx]
temp_classes = [class_ids[i] for i in temp_idx]

# ==== Step 2.5: Filter temp_files to remove classes with <2 samples ====
temp_class_counts = Counter(temp_classes)
filtered_temp_files = []
filtered_temp_classes = []

for f, c in zip(temp_files, temp_classes):
    if temp_class_counts[c] > 1:
        filtered_temp_files.append(f)
        filtered_temp_classes.append(c)

print(f"⚠️ Removed {len(temp_files) - len(filtered_temp_files)} temp files with rare classes")

# ==== Step 2.6: Stratified split of temp into val and test ====
splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(splitter2.split(filtered_temp_files, filtered_temp_classes))

val_files = [filtered_temp_files[i] for i in val_idx]
test_files = [filtered_temp_files[i] for i in test_idx]

# ==== Step 3: Copy files ====
def copy_all_formats(files, split):
    for file in files:
        for ext in ['.jpg', '.png']:
            src_img = os.path.join(images_path, file + ext)
            if os.path.exists(src_img):
                break
        else:
            print(f"⚠️ Image file missing for {file}")
            continue

        # Copy image
        shutil.copy(src_img, os.path.join(base_path, split, "images", os.path.basename(src_img)))

        # Copy label formats
        for lbl_dir, subfolder in [
            (labels_path, "labels"),
            (aa_labels_path, "aa_labels"),
            (dota_labels_path, "dota_format_labels"),
        ]:
            src_lbl = os.path.join(lbl_dir, file + ".txt")
            dst_lbl = os.path.join(base_path, split, subfolder, file + ".txt")
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)
            else:
                print(f"⚠️ Missing {subfolder} for {file}")

copy_all_formats(train_files, "train")
copy_all_formats(val_files, "val")
copy_all_formats(test_files, "test")


# ==== Final Summary ====
print(f"\n✅ Stratified split complete:")
print(f"  Train: {len(train_files)}")
print(f"  Val:   {len(val_files)}")
print(f"  Test:  {len(test_files)}")


def count_classes_in_split(split_dir):
    label_dir = os.path.join(base_path, split_dir, "labels")
    class_counter = Counter()

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counter[class_id] += 1
    return class_counter

print("\n📊 Class distribution per split:")
for split in ["train", "val", "test"]:
    counter = count_classes_in_split(split)
    print(f"  {split.capitalize()}:")
    for cls, count in sorted(counter.items()):
        print(f"    Class {cls}: {count}")