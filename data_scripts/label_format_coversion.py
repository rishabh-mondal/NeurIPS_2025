import os
from PIL import Image

## OBB->aa
import os
import glob

# Constants
IMG_WIDTH = 128
IMG_HEIGHT = 128

# Paths
base_label_path = "./labels"
aa_label_path = os.path.join(os.path.dirname(base_label_path), "aa_labels")
print(f"Base label path: {base_label_path}")
print(f"AA label path: {aa_label_path}")
os.makedirs(aa_label_path, exist_ok=True)

def convert_obb_to_aabb(coords):
    xs = coords[::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    if width < 0 or height < 0:
        raise ValueError("Width and height must be positive values.")

    return [x_center, y_center, width, height]

# Process each label file
label_files = glob.glob(os.path.join(base_label_path, "*.txt"))

for label_file in label_files:
    with open(label_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 9:
            print(f"Skipping malformed line in {label_file}: {line}")
            continue
        class_id = parts[0]
        coords = list(map(float, parts[1:]))
        try:
            xc, yc, w, h = convert_obb_to_aabb(coords)
        except Exception as e:
            print(f"Error in {label_file}: {e}")
            continue

        new_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # Write to new file
    out_path = os.path.join(aa_label_path, os.path.basename(label_file))
    with open(out_path, 'w') as out_f:
        out_f.write('\n'.join(new_lines))

print(f"✅ Converted {len(label_files)} files and saved to: {aa_label_path}")

## OBB -> DOTA Format
import os
from tqdm import tqdm

# Mapping from class ID to class name
class_mapping = {0: "CFCBK", 1: "FCBK", 2: "Zigzag"}

def convert_to_dota_format(input_path, output_path, image_width, image_height):
    """
    Convert a single label file from normalized format (YOLO-OBB style)
    to DOTA format.
    """
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            values = line.strip().split()
            if len(values) < 9:
                continue  # Skip incomplete lines

            class_id = int(values[0])
            coordinates = list(map(float, values[1:9]))

            # Denormalize coordinates
            denormalized_coords = [
                round(coordinates[i] * (image_width if i % 2 == 0 else image_height), 2)
                for i in range(8)
            ]

            class_name = class_mapping.get(class_id, "UNKNOWN")
            # Format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
            dota_line = " ".join(map(str, denormalized_coords)) + f" {class_name} 0\n"
            outfile.write(dota_line)

def process_folder(input_folder, output_folder, image_width, image_height):
    """
    Process all `.txt` label files in the input folder and convert to DOTA format.
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' not found")
    
    os.makedirs(output_folder, exist_ok=True)
    label_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".txt")])

    for file_name in tqdm(label_files, desc="Converting to DOTA format", unit="file"):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        convert_to_dota_format(input_path, output_path, image_width, image_height)

# === Main Call ===
if __name__ == "__main__":
    input_folder = "./labels"
    output_folder = os.path.join(os.path.dirname(input_folder), "dota_labels")
    image_width = 128
    image_height = 128

    process_folder(input_folder, output_folder, image_width, image_height)

    print(f"\n✅ All label files converted and saved to: {output_folder}")

