
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

import os
import requests
import ee
import rasterio
from rasterio.errors import RasterioIOError
from tqdm import tqdm

region="XXXXXX" # Specify the region you want to download tiles for
save_path=f"./sentinel_tiles/{region}"
os.makedirs(save_path, exist_ok=True)


filtered_tiles_path=f"./sentinel_metadata/{region}_missing_tiles.geojson"
# Stats
filtered_tiles_gdf = gpd.read_file(filtered_tiles_path)

num_tiles = len(filtered_tiles_gdf)
average_file_size_mb = 1.80
total_file_size_mb = num_tiles * average_file_size_mb
total_memory_gb = total_file_size_mb / 1024

print(f"Total number of filtered tiles: {num_tiles}")
print(f"Estimated total file size: {total_file_size_mb:.2f} MB")
print(f"Estimated total memory required: {total_memory_gb:.2f} GB")

import ee

ee.Initialize()

# Function to mask clouds
def mask_clouds(image):
    qa = image.select('QA60').uint16()  # Ensure it's an integer
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)  # Identify non-cloud pixels
    return image.updateMask(cloud_mask)

# Function to download a single tile
def download_tile(tile_geometry, tile_name, save_path):
    try:
        ee_tile = ee.Geometry.Polygon(list(tile_geometry.exterior.coords))

        # Get the best available image
        image_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(ee_tile) \
            .filterDate('2024-01-01', '2025-02-28') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1)) \
            .select(['B4', 'B3', 'B2', 'QA60']) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')

        # Try the least cloudy image first
        best_image = image_collection.first().clip(ee_tile)
        best_image = mask_clouds(best_image)

        # Use median image as backup
        backup_image = image_collection.median().clip(ee_tile)
        backup_image = mask_clouds(backup_image)

        # Compute valid pixel counts
        valid_pixels_best = ee.Number(best_image.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=ee_tile,
            scale=10
        ).values().get(0))

        valid_pixels_backup = ee.Number(backup_image.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=ee_tile,
            scale=10
        ).values().get(0))

        # Select the image with more valid pixels
        final_image = ee.Algorithms.If(valid_pixels_best.gt(valid_pixels_backup), best_image, backup_image)

        # Generate download URL
        url = ee.Image(final_image).getDownloadURL({
            'scale': 10,
            'region': ee_tile.getInfo(),
            'format': 'GEO_TIFF'
        })

        # Download the image
        response = requests.get(url)
        with open(f'{save_path}/{tile_name}.tif', 'wb') as f:
            f.write(response.content)

        print(f"Downloaded {tile_name}")

    except Exception as e:
        print(f"Skipping {tile_name}: {str(e)}")

# Function to check if a downloaded .tif file is valid
def check_tif_file(file_path, min_size_mb=1):
    """Check if the downloaded .tif file exists, is not empty, and is a valid GeoTIFF."""
    if not os.path.exists(file_path):
        return False

    # Check if the file size is greater than a reasonable threshold (e.g., 1 MB)
    if os.path.getsize(file_path) < min_size_mb * 1024 * 1024:  # 1 MB threshold
        print(f"File is too small: {file_path}")
        return False

    # Check if the file is a valid GeoTIFF using rasterio
    try:
        with rasterio.open(file_path) as ds:
            if ds.count < 1:
                print(f"Invalid raster (no bands): {file_path}")
                return False
        return True
    except rasterio.errors.RasterioIOError:
        print(f"Invalid GeoTIFF file: {file_path}")
        return False

# Function to handle download and validation sequentially
def download_and_validate_tile(tile_geometry, tile_name, save_path):
    try:
        ee.Initialize()  # Ensure Earth Engine is initialized in this subprocess
    except Exception as e:
        print(f"GEE initialization failed in subprocess for {tile_name}: {e}")
        return
    
    tile_path = os.path.join(save_path, f'{tile_name}.tif')

    # Download the tile
    download_tile(tile_geometry, tile_name, save_path)
    
    # Check if the downloaded file is valid
    if check_tif_file(tile_path):
        print(f"Tile {tile_name} is valid.")
    else:
        print(f"Tile {tile_name} is invalid, re-downloading...")
        download_tile(tile_geometry, tile_name, save_path)
        if check_tif_file(tile_path):
            print(f"Tile {tile_name} re-downloaded and is now valid.")
        else:
            print(f"Tile {tile_name} is still invalid after re-downloading.")

    

from joblib import Parallel, delayed

def process_tiles_parallel(filtered_tiles_gdf, save_path, n_jobs=4):
    Parallel(n_jobs=n_jobs)(
        delayed(download_and_validate_tile)(
            row.geometry, row.tile_name, save_path
        ) for _, row in filtered_tiles_gdf.iterrows()
    )
process_tiles_parallel(filtered_tiles_gdf, save_path, n_jobs=16)

print("All tiles processed.")       
# Run the tile processing
# process_tiles_sequentially(filtered_tiles_gdf, save_path)

tile_dir = save_path
valid_files = []
invalid_files = []

# Iterate through the downloaded files and check their validity
for filename in os.listdir(save_path):
    if filename.endswith('.tif'):
        file_path = os.path.join(save_path, filename)
        if check_tif_file(file_path):
            valid_files.append(filename)
        else:
            invalid_files.append(filename)

print(f"Valid files: {len(valid_files)}")
print(f"Invalid files: {len(invalid_files)}")
