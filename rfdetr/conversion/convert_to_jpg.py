# Convert directories of images from PNG and TIFF to JPG
# TODO: Update this code to use Multi-threading for faster conversion

import os
from PIL import Image
from tqdm import tqdm

# Set your source and output folder paths here
source_folders = ["/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/rfdetr/data/train/images_png",
                    "/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/rfdetr/data/val/images_png",
                    "/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/rfdetr/data/test/images_png"]

output_folders = ["/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/rfdetr/data/train/images",
                    "/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/rfdetr/data/val/images",
                    "/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/rfdetr/data/test/images"]


# Ensure the number of source and output folders match
if len(source_folders) != len(output_folders):
    raise ValueError("The number of source folders must match the number of output folders.")

# Process each folder pair
for source_folder, output_folder in zip(source_folders, output_folders):
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Get all PNG and TIFF files in the source folder
        image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.tiff', '.tif'))]

        # Convert each image to JPG
        for filename in tqdm(image_files, desc=f"Processing {source_folder}"):
            try:
                image_path = os.path.join(source_folder, filename)
                jpg_filename = os.path.splitext(filename)[0] + '.jpg'
                jpg_path = os.path.join(output_folder, jpg_filename)

                with Image.open(image_path) as img:
                    # Convert to RGB to avoid issues with alpha channels
                    rgb_img = img.convert('RGB')
                    rgb_img.save(jpg_path, 'JPEG')

                # print(f"Converted: {filename} -> {jpg_filename}")
            except Exception as e:
                print(f"❌ Error converting {filename}: {e}")

    except Exception as e:
        print(f"❌ Error processing folder {source_folder}: {e}")

print("✅ All PNG and TIFF images have been converted to JPG.")
