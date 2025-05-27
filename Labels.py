import json
from shapely import geometry
from pimsys.regions.RegionsDb import RegionsDb
import rasterio.features
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from geopy.distance import geodesic
import requests
import cv2
from PIL import Image
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import os

def get_utc_timestamp(x: datetime):
        return int((x - datetime(1970, 1, 1)).total_seconds())


def get_images(database, config, region, interval_utc):
    coords = [region['bounds'].centroid.x, region['bounds'].centroid.y]
    images = database.get_optical_images_containing_point_in_period(coords, interval_utc)

    wms_images = sorted(images, key=lambda x: x["capture_timestamp"])
    wms_images = [x for x in wms_images if "Sentinel" not in x["source"]]
    all_image_ids = []
    all_images = []

    for image in wms_images:
        if image["wms_layer_name"] not in all_image_ids:
            all_image_ids.append(image["wms_layer_name"])
            all_images.append(image)
    # Sort images based on classification
#    return all_images
    return all_images

def get_image_from_layer(layer, region_bounds):
    layer_name = layer['wms_layer_name']

    # Define layer name
    pixel_resolution_x = layer["pixel_resolution_x"]
    pixel_resolution_y = layer["pixel_resolution_y"]

    region_width = geodesic((region_bounds.bounds[1], region_bounds.bounds[0]), (region_bounds.bounds[1], region_bounds.bounds[2])).meters
    region_height = geodesic((region_bounds.bounds[1], region_bounds.bounds[0]), (region_bounds.bounds[3], region_bounds.bounds[0])).meters

    width = int(round(region_width / pixel_resolution_x))
    height = int(round(region_height / pixel_resolution_y))

    arguments = {
        'layer_name': layer_name,
        'bbox': '%s,%s,%s,%s' % (region_bounds.bounds[0], region_bounds.bounds[1], region_bounds.bounds[2], region_bounds.bounds[3]),
        'width': width,
        'height': height
    }

    # get URL
    if 'image_url' in layer.keys():
        if layer['downloader'] == 'geoserve':
            arguments['bbox'] = '%s,%s,%s,%s' % (region_bounds[1], region_bounds[0], region_bounds[3], region_bounds[2])
            url = layer['image_url'] + "&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&CRS=epsg:4326&BBOX=%(bbox)s&WIDTH=%(width)s&HEIGHT=%(height)s&FORMAT=image/png&LAYERS=%(layer_name)s" % arguments
        elif layer['downloader'] == 'sentinelhub':
            url = layer['image_url']
        else:
            url = layer['image_url'] + "&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&SRS=epsg:4326&BBOX=%(bbox)s&WIDTH=%(width)s&HEIGHT=%(height)s&FORMAT=image/png&LAYERS=%(layer_name)s" % arguments
    else:
        url = "https://maps.orbitaleye.nl/mapserver/?map=/maps/_%(layer_name)s.map&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&SRS=epsg:4326&BBOX=%(bbox)s&WIDTH=%(width)s&HEIGHT=%(height)s&FORMAT=image/png&LAYERS=%(layer_name)s" % arguments

    if layer['downloader'] == 'geoserve':
        resp = requests.get(url, auth=('ptt', 'yOju6YLPK6Pnqm2C'))
    else:
        resp = requests.get(url, auth=('mapserver_user', 'tL3n3uoPnD8QYiph'))
	
    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]]

    return image

def save_tif_coregistered(filename, image, poly, channels=3, factor=1):
    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                       poly.bounds[2], poly.bounds[3],
                                                       width/factor, height/factor)

    new_dataset = rasterio.open(filename, 'w', driver='GTiff',
                                height=height/factor, width=width/factor,
                                count=channels, dtype='uint8',
                                crs='+proj=latlong',
                                transform=geotiff_transform)

    # Write bands
    if channels>1:
     for ch in range(0, image.shape[2]):
       new_dataset.write(image[:,:,ch], ch+1)
    else:
       new_dataset.write(image, 1)
    new_dataset.close()
    # print('Image saved to', filename)
    return True

# Shapefile paths
shp1_path = "/cephfs/work/rithvik/OE_CL_shps/gas_transmission/gas_transmission.gt_building_exi_extent.shp"  # extent path
shp2_path = "/cephfs/work/rithvik/OE_CL_shps/gas_transmission/gas_transmission.gt_building_exi_location.shp" # location path

# Load shapefiles
gdf1 = gpd.read_file(shp1_path)
gdf2 = gpd.read_file(shp2_path)

gdf = pd.concat([gdf1,gdf2])

config = {
    "regions_db": {
        "host": "sar-ccd-db.orbitaleye.nl",
        "port": 5433,
        "user": "postgres",
        "password": "sarccd-db",
        "database": "sarccd2"
        }
    }

# customer = 'National-Fuel-2024'
# customer = 'National-Grid-2024'
customer = 'BHE-2024'

database = RegionsDb(config['regions_db'])
database_customer = database.get_regions_by_customer(customer)
#database.close()


dates = [datetime(2025, 2, 15), datetime(2025, 4, 1)]
interval_utc = [get_utc_timestamp(dates[0]), get_utc_timestamp(dates[1])]


# Select a region
for _, region in enumerate(tqdm(database_customer)):
        #print(region)
    # region = database_customer[:]

    images = get_images(database, config, region, interval_utc)


    #    print('len', len(images))
    if images:
        img = get_image_from_layer(images[-1], region['bounds'])
        # save_tif_coregistered("/home/rithvik/YOLO/image.tif", img, region['bounds'], channels=3, factor=1)
        #print(img.shape)
        img_pil = Image.fromarray(img)

        # Get image width and height
        img_width, img_height = img_pil.size

        # After loading shapefiles but before filtering

        # Print CRS information for debugging
        # print(f"Shapefile CRS: {gdf.crs}")
        # print(f"Region bounds type: {type(region['bounds'])}")
        # print(f"Region bounds: {region['bounds'].bounds}")

        # Check if any buildings exist in the shapefile
        # print(f"Total buildings in shapefile: {len(gdf)}")

        # First, create the region_gdf with the CORRECT initial CRS (EPSG:4326)
        region_gdf = gpd.GeoDataFrame({'geometry': [region['bounds']]}, crs="EPSG:4326")

        # Convert both to the same CRS (Web Mercator)
        gdf = gdf.to_crs("EPSG:3857")
        region_gdf = region_gdf.to_crs("EPSG:3857")

        # # Add a debug check with a buffer to see if there are buildings in the vicinity
        # x_min, y_min, x_max, y_max = region_gdf.geometry[0].bounds
        # buffer_dist = 5000  # buffer in meters (5km)
        # larger_bounds = geometry.box(x_min-buffer_dist, y_min-buffer_dist, 
        #                         x_max+buffer_dist, y_max+buffer_dist)
        # larger_gdf = gpd.GeoDataFrame({'geometry': [larger_bounds]}, crs="EPSG:3857")
        # nearby = gdf[gdf.intersects(larger_bounds)]
        # print(f"Buildings in 5km buffer: {len(nearby)}")

        # Now perform the spatial intersection with the correctly transformed geometry
        buildings_gdf = gdf[gdf.intersects(region_gdf.geometry[0])]
        # print(f"Number of buildings found: {len(buildings_gdf)}")

        # Create a mask from the filtered GeoDataFrame
        if not buildings_gdf.empty:
            # Prepare bounding box data for saving
            bbox_data = []

            # Iterate through each building in the filtered GeoDataFrame
            for index, row in buildings_gdf.iterrows():
                # Get the geometry of the building
                geom = row['geometry']

                # Check if the geometry is a Point
                if geom.geom_type == 'Point':
                    # Define the side length of the square bounding box (adjust as needed)
                    square_size_meters = 2  # e.g., 5 meters

                    # Create a square bounding box around the point
                    point_x, point_y = geom.x, geom.y
                    minx = point_x - square_size_meters / 2
                    miny = point_y - square_size_meters / 2
                    maxx = point_x + square_size_meters / 2
                    maxy = point_y + square_size_meters / 2
                    bbox = (minx, miny, maxx, maxy)
                else:
                    # Get the bounding box of the building
                    bbox = row['geometry'].bounds  # (minx, miny, maxx, maxy)

            # Convert bounding box coordinates to pixel coordinates
                minx, miny, maxx, maxy = bbox

                
                region_bounds = region_gdf.geometry[0].bounds  # Get bounds from the transformed geometry
                region_minx, region_miny, region_maxx, region_maxy = region_bounds

                # Calculate normalized coordinates directly (don't convert to pixels first)
                x_center = ((minx + maxx) / 2 - region_minx) / (region_maxx - region_minx)
                y_center = ((miny + maxy) / 2 - region_miny) / (region_maxy - region_miny) 
                # Note: Y-coordinates in YOLO are measured from top to bottom, so invert if needed
                y_center = 1 - y_center  # Only if you need to flip the y-axis

                width = (maxx - minx) / (region_maxx - region_minx)
                height = (maxy - miny) / (region_maxy - region_miny)

                # Validation - make sure values are between 0 and 1
                # if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1 or width < 0 or width > 1 or height < 0 or height > 1:
                    # print(f"Warning: Out of bounds normalized coordinates: {x_center}, {y_center}, {width}, {height}")
                    # Skip this box or clamp values

                # Append the bounding box data (change class_id as needed)
                bbox_data.append(f"0 {x_center} {y_center} {width} {height}")

            # Save the bounding box data to a text file
            output_path = '/cephfs/work/rithvik/datasets/datasets/BHE/test/2025Q1/labels/{}_{}.txt'.format(region['id'], region['region_customer_id'])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write('\n'.join(bbox_data))

            # Save the image as GeoTIFF instead of PNG
            image_output_path = '/cephfs/work/rithvik/datasets/datasets/BHE/test/2025Q1/images/{}_{}.tif'.format(region['id'], region['region_customer_id'])
            os.makedirs(os.path.dirname(image_output_path), exist_ok=True)

            # Use save_tif_coregistered to preserve geographic information
            save_tif_coregistered(
                image_output_path,  # Output path with .tif extension
                img,                # The image array (not the PIL image)
                region['bounds'],   # The geographic bounds
                channels=3,         # RGB image
                factor=1            # No downsampling
            )
        # else:
        #     print(f"No buildings found in region {region['id']}")

# save_tif_coregistered("/home/rithvik/YOLO/image.tif", images, region['bounds'], channels=3, factor=1)