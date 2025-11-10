from pathlib import Path
import geopandas as gpd
import pandas as pd

def flagging(img_path, new_buildings, removed_buildings, total_buildings, criteria=0.3, simple=False):
    
    """
    Flags images based on the ratio of new and removed buildings to total buildings.
    
    Parameters:
    - new_buildings: List of new building features
    - removed_buildings: List of removed building features
    - total_buildings: Total number of buildings in the baseline
    - criteria: Threshold ratio to flag an image (default is 0.3)
    - simple: If True, uses combined ratio of new and removed buildings; 
              if False, uses building density based criteria.
    
    Returns:
    - flag: Boolean indicating if the image is flagged (True) or not (False)
    """

    valid_areas_path = Path('/cephfs/work/rithvik/datasets/datasets/BHE/valid_areas/')
    for valid_path in valid_areas_path.glob("*.json"):
        if img_path.stem in valid_path.stem:
            valid_area_path = valid_path
            break
    
    # Load valid area polygon
    valid_area_gdf = None
    if valid_area_path and valid_area_path.exists():
        valid_area_gdf = gpd.read_file(valid_area_path)
    
    # Calculate area of the valid area polygon
    if valid_area_gdf is not None:
        # Convert to appropriate UTM projection for accurate area calculation
        valid_area_projected = valid_area_gdf.to_crs(valid_area_gdf.estimate_utm_crs())
        area_sq_meters = valid_area_projected.area[0]
        area_sq_km = area_sq_meters / 1_000_000
    
    new_count = len(new_buildings)
    removed_count = len(removed_buildings)
    if simple:
        if total_buildings > 0:
            ratio = (new_count + removed_count) / total_buildings
        else:
            ratio = 0

        flag = (ratio >= criteria)
    else:
        base_area  = 0.5625 # Estimated area of image that is 1500x1500 pixels at 0.5m resolution
        if total_buildings > int(10 * area_sq_km / base_area):
            ratio = (new_count + removed_count) / total_buildings
        else:
            ratio = 0

        flag = (ratio >= criteria)
    
    return flag


