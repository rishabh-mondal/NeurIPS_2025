
import os
import ee
import geemap
import requests
import leafmap
import numpy as np
import pandas as pd
import geopandas as gpd
import concurrent.futures
import matplotlib.pyplot as plt
from shapely.geometry import box  
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import ee
import requests
from tqdm import tqdm
import ast

from rasterio.errors import RasterioIOError

import rasterio

from dask import delayed, compute
from dask.diagnostics import ProgressBar
from tqdm import tqdm


region = "XXXX"#specify the region you want to download tiles for


# Read metadata
metadata_path = f"./sentinel_metadata/{region}_sentinel_metadata.geojson"
tif_dir = f"./sentinel_tiles/{region}/"
gdf = gpd.read_file(metadata_path)
print(gdf.crs)

## visualization of tiles


def parse_center(coord_str):
    try:
        return ast.literal_eval(coord_str)
    except Exception:
        return (None, None)
gdf["parsed_center"] = gdf["center_coordinates"].apply(parse_center)

# Check if area_m2 exists; if not, compute it
if 'area_m2' not in gdf.columns:
    gdf['area_m2'] = gdf.to_crs("EPSG:3857").geometry.area

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
gdf.plot(ax=ax,edgecolor='black', facecolor='none')


## Tile Devide into patches


patch_size = 128
overlap = 30
stride = patch_size - overlap

# === Load tile metadata ===
gdf = gpd.read_file(metadata_path)
gdf = gdf.to_crs("EPSG:4326")

def parse_center(coord_str):
    try:
        return ast.literal_eval(coord_str)
    except:
        return (None, None)

gdf["parsed_center"] = gdf["center_coordinates"].apply(parse_center)

# === Patch Generation & Visualization ===
patch_records = []
tif_files = sorted([f for f in os.listdir(tif_dir) if f.endswith(".tif")])  

for tif_name in tqdm(tif_files):
    tif_path = os.path.join(tif_dir, tif_name)

    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height
        transform = src.transform

        x_centers = []
        x = 0
        while x + patch_size <= width:
            x_centers.append(x + patch_size // 2)
            x += stride
        if x_centers[-1] + patch_size // 2 < width:
            x_centers.append(width - patch_size // 2)

        y_centers = []
        y = 0
        while y + patch_size <= height:
            y_centers.append(y + patch_size // 2)
            y += stride
        if y_centers[-1] + patch_size // 2 < height:
            y_centers.append(height - patch_size // 2)

        # # === Read RGB bands ===
        # img = src.read([1, 2, 3])
        # img = np.transpose(img, (1, 2, 0))
        # img = (img - img.min()) / (img.max() - img.min())

        # === Plot Full Tile with Patches ===
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax.imshow(img)
        # ax.set_title(f"All Patches over Tile: {tif_name}")

        for y in y_centers:
            for x in x_centers:
                x0 = x - patch_size // 2
                y0 = y - patch_size // 2
                x1 = x0 + patch_size
                y1 = y0 + patch_size

                # Bounding box in lon/lat
                # Bounding box in lon/lat
                lon_min, lat_max = rasterio.transform.xy(transform, y0, x0, offset='ul')
                lon_max, lat_min = rasterio.transform.xy(transform, y1, x1, offset='lr')
                lon_center = (lon_min + lon_max) / 2
                lat_center = (lat_min + lat_max) / 2

                poly = box(lon_min, lat_min, lon_max, lat_max)

                patch_records.append({
                    "tile_name": tif_name,
                    "lon_center": lon_center,
                    "lat_center": lat_center,
                    "geometry": poly
                })

# # === Save Patch Metadata ===
patches_gdf = gpd.GeoDataFrame(patch_records, crs="EPSG:4326")
patches_gdf.to_file(f"./sentinel/{region}_metadata.geojson", driver="GeoJSON")
# patches_gdf[["tile_name", "lon_center", "lat_center"]].to_csv(f"{region}_patch_centers.csv", index=False)

print(f"Saved {len(patches_gdf)} patches.")

from joblib import Parallel, delayed
import os
import ast
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.errors import RasterioIOError
from shapely.geometry import box
from tqdm import tqdm
from PIL import Image


# Setup
patch_save_path = "./sentinel"
save_dirs = {
    "rgb": f"{patch_save_path}/{region}/rgb",
}
for path in save_dirs.values():
    os.makedirs(path, exist_ok=True)

# Load metadata
gdf = gpd.read_file(metadata_path).to_crs("EPSG:4326")
gdf["parsed_center"] = gdf["center_coordinates"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else (None, None))

# Normalization function
def norm(band):
    return ((band - band.min()) / (band.max() - band.min() + 1e-5) * 255).astype(np.uint8)

# === Parallelized Tile Processor ===
def process_tile(tif_name):
    tif_path = os.path.join(tif_dir, tif_name)
    local_records = []

    try:
        with rasterio.open(tif_path) as src:
            width, height = src.width, src.height
            transform = src.transform

            x_centers = [x + patch_size // 2 for x in range(0, width - patch_size + 1, stride)]
            if x_centers and x_centers[-1] + patch_size // 2 < width:
                x_centers.append(width - patch_size // 2)

            y_centers = [y + patch_size // 2 for y in range(0, height - patch_size + 1, stride)]
            if y_centers and y_centers[-1] + patch_size // 2 < height:
                y_centers.append(height - patch_size // 2)

            for yc in y_centers:
                for xc in x_centers:
                    x0, y0 = xc - patch_size // 2, yc - patch_size // 2
                    x1, y1 = x0 + patch_size, y0 + patch_size

                    lon_min, lat_max = rasterio.transform.xy(transform, y0, x0, offset='ul')
                    lon_max, lat_min = rasterio.transform.xy(transform, y1, x1, offset='lr')
                    lon_center = round((lon_min + lon_max) / 2, 4)
                    lat_center = round((lat_min + lat_max) / 2, 4)

                    poly = box(lon_min, lat_min, lon_max, lat_max)
                    local_records.append({
                        "tile_name": tif_name,
                        "lon_center": lon_center,
                        "lat_center": lat_center,
                        "geometry": poly
                    })

                    try:
                        red = src.read(1)[y0:y1, x0:x1]
                        green = src.read(2)[y0:y1, x0:x1]
                        blue = src.read(3)[y0:y1, x0:x1]
                    except:
                        continue

                    red_norm = norm(red)
                    green_norm = norm(green)
                    blue_norm = norm(blue)

                    composites = {
                        "rgb": np.dstack((red_norm, green_norm, blue_norm))
                    }

                    patch_id = f"{lat_center:.4f}_{lon_center:.4f}"
                    for name, patch_img in composites.items():
                        save_path = os.path.join(save_dirs[name], f"{patch_id}.png")
                        # save the images in rgb format
                        Image.fromarray(patch_img).convert("RGB").save(save_path)


    except RasterioIOError:
        print(f"Skipped unsupported file: {tif_path}")

    return local_records


potential_image_shape = gpd.read_file(f"./sentinel/{region}_metadata.geojson")
print(potential_image_shape.head(2))
print("Original CRS:", potential_image_shape.crs)
potential_image_shape = potential_image_shape.to_crs("EPSG:3857")
print("Updated CRS:", potential_image_shape.crs)


# Parallel execution
tif_files = sorted([f for f in os.listdir(tif_dir) if f.endswith(".tif")])
all_records = Parallel(n_jobs=42)(delayed(process_tile)(tif_name) for tif_name in tqdm(tif_files, desc="Processing Sentinel Tiles"))
patch_records = [record for sublist in all_records for record in sublist]  # Flatten

## Label Processing
statename="XXXX" #specify the state name
label_path=f"./{statename}.geojson"
gdf_labels = gpd.read_file(label_path).drop("style", errors="ignore", axis=1)

color_mapping = {"CFCBK": "red", "FCBK": "orange", "Zigzag": "green"}
gdf_labels["style"] = gdf_labels["class_name"].apply(lambda x: {"color": color_mapping[x]})
gdf_labels_webm=gdf_labels.to_crs(potential_image_shape.crs)
gdf_labels_webm.reset_index(inplace=True, drop=True)
print(gdf_labels_webm.crs)
print("Number of labels:", len(gdf_labels_webm))

# gdf_labels_webm.head(2)
gdf_labels_webm=gdf_labels.to_crs(potential_image_shape.crs)
gdf_labels_webm.reset_index(inplace=True, drop=True)
print(gdf_labels_webm.crs)
print("Number of labels:", len(gdf_labels_webm))

images_with_label=gpd.sjoin(potential_image_shape,gdf_labels_webm,predicate="contains")
images_with_label['geometry_right'] = images_with_label['index_right'].apply(lambda x: gdf_labels_webm.loc[x, 'geometry'])
print(f"Number of labels to write: {len(images_with_label)}")
print(f"Number of unique images: {len(images_with_label.drop_duplicates(subset='geometry'))}")
print(f"Number of unique labels: {len(images_with_label.drop_duplicates(subset='geometry_right'))}")

gdf_labels_webm["geometry"]
class_mapping = {"CFCBK": 0, "FCBK": 1, "Zigzag": 2}
def get_yolo_label(x):
    image_polygon=x['geometry']
    label_polygon=x['geometry_right']
    min_x, min_y, max_x, max_y = image_polygon.bounds
    coords = np.array(label_polygon.__geo_interface__['coordinates'][0][:-1])  
    coords[:, 0] = (coords[:, 0] - min_x) / (max_x - min_x)
    coords[:, 1] = 1 - (coords[:, 1] - min_y) / (max_y - min_y)
    coords = coords.ravel()
    assert len(coords) == 8
    class_id = class_mapping[x['class_name']]
    label = np.zeros(9) * np.nan
    label[0] = class_mapping.get(x['class_name'],-1)
    label[1:] = coords
    return label

images_with_label['yolo_label_s'] = images_with_label.apply(get_yolo_label, axis=1)
print("Total geometries:", len(images_with_label['geometry_right']))
geometry_gdf = gpd.GeoDataFrame(geometry=images_with_label['geometry_right'],crs="EPSG:3857" )
geometry_gdf = geometry_gdf.to_crs("EPSG:4326")
m = leafmap.Map()
m.add_basemap("HYBRID")
m.add_gdf(geometry_gdf, layer_name="Detected Bounding Boxes",zoom_to_layer=True)    
m

def has_negative(label):
    return np.any(label < 0)

# Filter rows where any value in label is negative (excluding class_id if needed)
negatives = images_with_label[images_with_label['yolo_label_s'].apply(has_negative)]

print(f"Found {len(negatives)} entries with negative values in YOLO-OBB labels.")
if len(negatives) > 0:
    print(negatives[['lat_center', 'lon_center', 'yolo_label']].head())


from os.path import join
import glob

save_dir= f"./sentinel/{region}"

label_dir_path = join(save_dir, "labels")
print(f"Creating directory: {label_dir_path}")
os.makedirs(label_dir_path, exist_ok=True)
grouped = images_with_label.groupby(['lat_center', 'lon_center'])
for (lat_center, lon_center), group in grouped:
    labels = np.vstack(group['yolo_label_s'].values)  # stack all label rows
    save_path = join(label_dir_path, f"{lat_center:.4f}_{lon_center:.4f}.txt")
    np.savetxt(save_path, labels, fmt="%d %f %f %f %f %f %f %f %f")

print(f"Saved {len(grouped)} YOLO-OBB label files to: {label_dir_path}")

label_files = glob.glob(os.path.join(label_dir_path, "*.txt"))

bad_files = []
for file in label_files:
    data = np.loadtxt(file, ndmin=2)  
    if np.any(data[:, 1:] < 0): 
        bad_files.append(file)

print(f"Found {len(bad_files)} files with negative label coordinates.")
if bad_files:
    print("Examples:", bad_files[:5])



import os

label_dir = f"./sentinel/{region}/labels"
image_dir = f"./sentinel/{region}/rgb"

# Get filenames without extensions
label_files = set(os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".txt"))
image_files = set(os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".png"))

# Compute matches and mismatches
matched = label_files & image_files
unmatched_labels = label_files - image_files
unmatched_images = image_files - label_files

# Print counts
print(f"✅ Matched files: {len(matched)}")
print(f"❌ Unmatched labels (no corresponding image): {len(unmatched_labels)}")
print(f"❌ Unmatched images (no corresponding label): {len(unmatched_images)}")

# Optionally, print mismatched file names
# print("Unmatched label files:", unmatched_labels)
# print("Unmatched image files:", unmatched_images)

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from os.path import join, splitext, basename

# CONFIGURATION
label_dir = f"./sentinel/{region}/labels"
image_dir = f"./sentinel/{region}/rgb"
image_size = 128  # Sentinel image size in pixels
NUM_IMAGES_TO_SHOW = 150

# LOAD LABEL FILES
label_files = glob(join(label_dir, "*.txt"))
selected_label_files = np.random.choice(label_files, size=min(NUM_IMAGES_TO_SHOW, len(label_files)), replace=False)

# PLOT MULTIPLE IMAGES
cols = 6
rows = int(np.ceil(len(selected_label_files) / cols))
plt.figure(figsize=(cols * 5, rows * 5))

for i, label_file in enumerate(selected_label_files, start=1):
    base = splitext(basename(label_file))[0]
    image_file = join(image_dir, base + ".png")

    if not os.path.exists(image_file):
        print(f"🚫 Image not found: {image_file}")
        continue

    try:
        # Load image and labels
        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image)
        labels = np.loadtxt(label_file, ndmin=2)

        # Plot
        plt.subplot(rows, cols, i)
        plt.imshow(image_np)
        plt.axis("off")
        plt.title(f"{base}", fontsize=12)

        for label in labels:
            class_id = int(label[0])
            coords = label[1:] * image_size
            coords = coords.reshape(-1, 2)
            coords = np.vstack([coords, coords[0]])  # close polygon
            plt.plot(coords[:, 0], coords[:, 1], 'r-', linewidth=2)
            plt.text(coords[0, 0], coords[0, 1], f"{class_id}", color="yellow", fontsize=8)

    except Exception as e:
        print(f"⚠️ Error with file {label_file}: {e}")

plt.tight_layout()
plt.show()
