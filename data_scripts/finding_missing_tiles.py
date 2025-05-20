
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


# Replace with your actual region
region = "XXXX"#specify the region you want to download tiles for

# Read metadata
metadata_path = f"./sentinel_metadata/{region}_sentinel_metadata.geojson"
tif_dir = f"./sentinel_tiles/{region}/"
gdf = gpd.read_file(metadata_path)

import os
import rasterio
import geopandas as gpd
import ast
from shapely.geometry import box
from rasterio.errors import RasterioIOError
from tqdm import tqdm
import matplotlib.pyplot as plt

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

# === Generate patch polygons ===
all_patches = []
skipped_files_count = 0  # Counter for skipped files
skkipped_files = []  # List to store skipped files

for tif_name in tqdm(os.listdir(tif_dir)):
    if not tif_name.endswith(".tif"):
        continue

    tif_path = os.path.join(tif_dir, tif_name)

    try:
        with rasterio.open(tif_path) as src:
            width, height = src.width, src.height
            transform = src.transform

            # Ensure complete coverage with overlap
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

            for y in y_centers:
                for x in x_centers:
                    x0 = x - patch_size // 2
                    y0 = y - patch_size // 2
                    x1 = x0 + patch_size
                    y1 = y0 + patch_size

                    # Convert pixel coordinates to lat/lon bounds
                    lon_min, lat_max = rasterio.transform.xy(transform, y0, x0, offset='ul')
                    lon_max, lat_min = rasterio.transform.xy(transform, y1, x1, offset='lr')

                    poly = box(lon_min, lat_min, lon_max, lat_max)
                    all_patches.append(poly)

    except RasterioIOError as e:
        
        # print(f"⚠️ Skipping unsupported file: {tif_path} — {e}")
        skkipped_files.append(tif_path)
        skipped_files_count += 1  # Increment counter for skipped files

        continue
    except Exception as e:
        # print(f"⚠️ Unexpected error with file {tif_path}: {e}")
        skipped_files_count += 1  # Increment counter for skipped files
        skkipped_files.append(tif_path)
        continue

# === Create GeoDataFrame of patches ===
patches_gdf = gpd.GeoDataFrame(geometry=all_patches, crs="EPSG:4326")

# === Plot patches over original tiles ===
fig, ax = plt.subplots(1, 1, figsize=(30, 30))
gdf.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1, label="Original Tiles")
patches_gdf.plot(ax=ax, edgecolor='blue', facecolor='none', linestyle='--', label="Patches")

# Annotate center coordinates
for _, row in gdf.iterrows():
    lon, lat = row["parsed_center"]
    if lon and lat:
        label = f"{lon:.2f}, {lat:.2f}"
        ax.annotate(label, (lon, lat), fontsize=6, color="white",
                    ha="center", va="center", bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"))

plt.title(f"{region} - Patch Overlay with {overlap}px Overlap")
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig(f"{region}_patch_overlay.png", dpi=300)

# Print the count of skipped files
print(f"Number of files skipped due to errors: {skipped_files_count}")
print("Skipped files:")
print(len(skkipped_files))

# === Configuration ===
metadata_path = f"./sentinel_metadata/{region}_sentinel_metadata.geojson"
tif_dir = f"./sentinel_tiles/{region}/"
output_geojson_path = f"./sentinel_metadata/{region}_missing_tiles.geojson"

# extract the file names from the skipped files lsit
skipped_tile_names = [os.path.basename(file) for file in skkipped_files]
print(len(skipped_tile_names))
print("Skipped tile names:", skipped_tile_names)
stripped_tile_names = [name.replace('.tif', '') for name in skipped_tile_names]
print("Stripped tile names:", stripped_tile_names)


gdf = gpd.read_file(metadata_path)
gdf = gdf.to_crs("EPSG:4326")
print(gdf.head(5))
# === Inspect column names ===
print("Columns in GeoDataFrame:", gdf.columns)
missing_tiles_gdf = gdf[gdf['tile_name'].isin(stripped_tile_names)]

# Replace 'file_name' with the actual column name if different
print(f"Found {len(missing_tiles_gdf = gdf[gdf['tile_name'].isin(stripped_tile_names)])} missing_tiles_gdf = gdf[gdf['tile_name'].isin(stripped_tile_names)] tiles.")
print(missing_tiles_gdf = gdf[gdf['tile_name'].isin(stripped_tile_names)].head())
# Print sample
missing_tiles_gdf = gdf[gdf['tile_name'].isin(stripped_tile_names)].to_file(output_geojson_path, driver="GeoJSON")