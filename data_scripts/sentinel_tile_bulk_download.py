
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
from joblib import Parallel, delayed
import os
import requests
import ee
import rasterio
from rasterio.errors import RasterioIOError
from tqdm import tqdm


region="XXXXXX" # Specify the region name here
# Set up the save path
save_path=f"./{region}"
os.makedirs(save_path, exist_ok=True)
# Set up the region of interest
# Load the shapefile
data_path = f"./shapes/{region}.geojson"

gdf=gpd.read_file(data_path)
gdf_projected = gdf.to_crs(epsg=32643)
gdf['area_m2'] = gdf_projected.area
gdf['area_km2'] = gdf['area_m2'] / (1000**2)
print(f"Area in square meters: {gdf['area_m2'].iloc[0]:,.2f} m²")
print(f"Area in square kilometers: {gdf['area_km2'].iloc[0]:,.2f} km²")


# bounds of the AOI (minx, miny, maxx, maxy)
minx, miny, maxx, maxy = gdf.total_bounds
grid_size = "XXXX"  # Specify the grid size here (in degrees)
lons = np.arange(minx, maxx, grid_size)
lats = np.arange(miny, maxy, grid_size)
tiles = []
center_coords = []
for lon in lons:
    for lat in lats:
        tile_geometry = box(lon, lat, lon + grid_size, lat + grid_size)
        tile = gpd.GeoDataFrame({'geometry': [tile_geometry]}, crs='EPSG:4326')
        tiles.append(tile)
        center_lon = lon + grid_size / 2
        center_lat = lat + grid_size / 2
        center_coords.append((center_lon, center_lat))

tiles_gdf = gpd.GeoDataFrame(pd.concat(tiles, ignore_index=True), crs='EPSG:4326')
tiles_gdf['center_coordinates'] = center_coords


# Create tiles
tiles = []
tiles_names = []
center_coords = []

for lon in lons:
    for lat in lats:
        tile_geometry = box(lon, lat, lon + grid_size, lat + grid_size)
        tile = gpd.GeoDataFrame({'geometry': [tile_geometry]}, crs='EPSG:4326')
        tiles.append(tile)

        # Create tile name and center coordinates
        tile_name = f"Tile_{lon:.4f}_{lat:.4f}"
        tiles_names.append(tile_name)

        center_lon = lon + grid_size / 2
        center_lat = lat + grid_size / 2
        center_coords.append((center_lon, center_lat))

print(f"Generated {len(tiles)} tiles.")

# Combine into a single GeoDataFrame
tiles_gdf = gpd.GeoDataFrame(pd.concat(tiles, ignore_index=True), crs='EPSG:4326')
tiles_gdf['tile_name'] = tiles_names
tiles_gdf['center_coordinates'] = center_coords
tiles_gdf['area_m2'] = tiles_gdf.geometry.area

# Area of AOI
aoi_area = gdf.area.sum()
print(f"Area of AOI: {aoi_area:.2f} square meters")

# Extract tile center and corners
def extract_tile_details(gdf):
    tile_details = []
    for _, row in gdf.iterrows():
        bounds = row.geometry.bounds
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        corners = {
            'top_left': (bounds[0], bounds[3]),
            'top_right': (bounds[2], bounds[3]),
            'bottom_left': (bounds[0], bounds[1]),
            'bottom_right': (bounds[2], bounds[1])
        }
        tile_details.append({
            'tile_name': row.tile_name,
            'center_coordinates': (center_x, center_y),
            'corner_coordinates': corners
        })
    return tile_details

tile_details_df = pd.DataFrame(extract_tile_details(tiles_gdf))

# print(tile_details_df)

# Sentinel-2 band info
sentinel2_bands = {
    "B2": {"name": "Blue", "resolution": 10, "central_wavelength": 490},
    "B3": {"name": "Green", "resolution": 10, "central_wavelength": 560},
    "B4": {"name": "Red", "resolution": 10, "central_wavelength": 665},
}

# Add band info to tiles
for band_name, band_info in sentinel2_bands.items():
    tile_details_df[f"{band_name}_resolution"] = band_info['resolution']
    tile_details_df[f"{band_name}_central_wavelength"] = band_info['central_wavelength']

# Intersection and filtering
tiles_gdf['intersection_area'] = tiles_gdf.intersection(gdf.unary_union).area
threshold_area = 0.015 * grid_size * grid_size
filtered_tiles_gdf = tiles_gdf[tiles_gdf['intersection_area'] > threshold_area]
print(f"Filtered {len(filtered_tiles_gdf)} tiles based on intersection area.")

# print(f"Filtered tiles: {filtered_tiles_gdf[['tile_name', 'intersection_area']]}")

# Extract tile details again for filtered tiles
filtered_tile_details_df = pd.DataFrame(extract_tile_details(filtered_tiles_gdf))

# Add band info again
for band_name, band_info in sentinel2_bands.items():
    filtered_tile_details_df[f"{band_name}_resolution"] = band_info['resolution']
    filtered_tile_details_df[f"{band_name}_central_wavelength"] = band_info['central_wavelength']

# Merge filtered tile details with filtered tiles_gdf
filtered_tiles_gdf['tile_details'] = filtered_tile_details_df.apply(lambda row: {
    'tile_name': row.tile_name,
    'center_coordinates': row.center_coordinates,
    'corner_coordinates': row.corner_coordinates,
    'B2_resolution': row.B2_resolution,
    'B2_central_wavelength': row.B2_central_wavelength,
    'B3_resolution': row.B3_resolution,
    'B3_central_wavelength': row.B3_central_wavelength,
    'B4_resolution': row.B4_resolution,
    'B4_central_wavelength': row.B4_central_wavelength,

}, axis=1)

# Save to GeoJSON
output_path = f"./sentinel_metadata/{region}_sentinel_metadata.geojson"   #metadata for sentinel tiles
filtered_tiles_gdf.to_crs('EPSG:4326').to_file(output_path, driver='GeoJSON')

# Stats
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

    # Clean up temporary files if needed


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
for filename in os.listdir(tile_dir):
    if filename.endswith('.tif'):
        file_path = os.path.join(tile_dir, filename)
        if check_tif_file(file_path):
            valid_files.append(filename)
        else:
            invalid_files.append(filename)

print(f"Valid files: {len(valid_files)}")
print(f"Invalid files: {len(invalid_files)}")
