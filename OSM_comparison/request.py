import requests
import json
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box
from shapely.errors import GEOSException
import warnings
import time
import os
import numpy as np
import pandas as pd
import folium
import traceback

# --- Configuration ---
SHAPEFILE_PATH = '/cephfs/work/rithvik/OE_CL_shps/gas_transmission/gas_transmission.gt_building_exi_location.shp'  # Location shapefile
MASK_SHAPEFILE_PATH = '/cephfs/work/rithvik/OE_CL_shps/gas_transmission/gas_transmission.gt_building_exi_extent.shp' # REQUIRED: Update this path! Building mask shapefile
FOUND_SHAPEFILE_PATH = '/cephfs/work/rithvik/OE_CL_shps/bhe_class_location_results_q1_2025/bhe_class_location_results.shp' # Shapefile containing buildings already found through class location
SHAPEFILE_FILTER_COLUMN = None  # Set to None if no filtering needed (applies to location shapefile)
SHAPEFILE_FILTER_VALUE = None  # Set to None if no filtering needed (applies to location shapefile)

# --- New Line Proximity Filter Config ---
LINE_SHAPEFILE_PATH = '/cephfs/work/rithvik/OE_CL_shps/pipeline.geojson'  # Set to None to disable line proximity filter
FILTER_DISTANCE_METERS = 214  # Distance (meters) around the line to keep buildings
# --- End New Config ---

MATCHING_DISTANCE_METERS = 30  # Max distance (meters) between centroids/points to consider a match (NOW USED AGAIN)
BUFFER_AREA_METERS = 214  # Extra margin around shapefile bounds for OSM query
OVERPASS_TIMEOUT_SECONDS = 180  # Timeout for the Overpass API request
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"

USER_AGENT = None #'BuildingComparisonScript/1.0 (user@email.com)' #<------Replace this line
TILE_SIZE_DEGREES = 0.1  # Approximate size of query tiles in degrees (adjust as needed)
TILE_REQUEST_DELAY_SECONDS = 1  # Delay between tile requests to be polite to API

# --- Define Output Directory ---
OUTPUT_DIR = "./tile_comparison_maps" # Replace with your desired output directory
OUTPUT_FILENAME_BASE = "overall_comparison_map.html"
OUTPUT_GEOJSON_BASE = "unmatched_osm_buildings.geojson"  # Changed base name
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_BASE)
OUTPUT_GEOJSON_FILENAME = os.path.join(OUTPUT_DIR, OUTPUT_GEOJSON_BASE)  # Uses updated base name

# --- Helper Function: Parse OSM Geometry from Overpass JSON ---
def parse_osm_geometry(element):
    """Converts OSM element 'geometry' (from out geom;) or lat/lon to Shapely object."""
    geom_data = element.get('geometry')
    elem_type = element.get('type')

    try:
        if geom_data:  # Using 'out geom;'
            if isinstance(geom_data, list):
                coords = [(p['lon'], p['lat']) for p in geom_data if 'lon' in p and 'lat' in p]
                if len(coords) >= 3 and coords[0] == coords[-1]:  # Closed way = Polygon
                    return Polygon(coords)
                elif len(coords) >= 2:  # Open way = LineString (less common for buildings)
                    return LineString(coords)
                elif len(coords) == 1:  # Single node from geom?
                    return Point(coords[0])
        elif elem_type == 'node' and 'lat' in element and 'lon' in element:  # Using 'out center;' or just node
            return Point(element['lon'], element['lat'])
        elif elem_type in ['way', 'relation'] and 'center' in element:  # Using 'out center;'
            center = element['center']
            return Point(center['lon'], center['lat'])
    except (GEOSException, ValueError, TypeError) as e:
        print(f"Warning: Could not parse geometry for element {element.get('type')}/{element.get('id')}: {e}")

    return None  # Return None if geometry can't be parsed

# --- Helper Function: Get OSM Data as GeoDataFrame ---
def get_osm_buildings_gdf(bounds_4326, api_url, timeout, user_agent):
    """Queries Overpass for buildings in bounds and returns a GeoDataFrame."""
    min_lon, min_lat, max_lon, max_lat = bounds_4326
    overpass_bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    overpass_query = f"""
    [out:json][timeout:{timeout}];
    (
      node["building"]({overpass_bbox});
      way["building"]({overpass_bbox});
      relation["building"]["type"="multipolygon"]({overpass_bbox});
    //node["addr:housenumber"]({overpass_bbox});
    );
    out geom;
    """

    print(f"Querying Overpass for buildings in bbox: {overpass_bbox}")
    payload = {'data': overpass_query}
    headers = {'User-Agent': user_agent}

    try:
        response = requests.post(api_url, data=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        osm_data = response.json()
    except requests.exceptions.Timeout:
        print("Error: Overpass API request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error querying Overpass API: {e}")
        print("Response text:", response.text if 'response' in locals() else "N/A")
        return None
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response from Overpass.")
        print("Raw response text:", response.text)
        return None

    print(f"Received {len(osm_data.get('elements', []))} elements from Overpass.")

    features = []
    for elem in osm_data.get('elements', []):
        geom = parse_osm_geometry(elem)
        if geom:
            props = elem.get('tags', {})
            props['osm_id'] = f"{elem.get('type')}/{elem.get('id')}"
            props['osm_type'] = elem.get('type')
            features.append({'geometry': geom, 'properties': props})

    if not features:
        print("Warning: No valid building geometries found in Overpass response.")
        return None

    try:
        gdf_osm = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        gdf_osm = gdf_osm[gdf_osm.geometry.type.isin(['Point', 'Polygon', 'MultiPolygon'])]
        print(f"Created OSM GeoDataFrame with {len(gdf_osm)} buildings (Points/Polygons).")
        return gdf_osm
    except Exception as e:
        print(f"Error creating GeoDataFrame from OSM features: {e}")
        return None

# --- Helper Function: Create Tile Bounding Boxes ---
def create_tile_bboxes(overall_bounds_4326, tile_size_degrees):
    """Divides an overall bounding box into smaller tile bounding boxes."""
    min_lon, min_lat, max_lon, max_lat = overall_bounds_4326
    lon_steps = np.arange(min_lon, max_lon, tile_size_degrees)
    lat_steps = np.arange(min_lat, max_lat, tile_size_degrees)

    tile_bboxes = []
    for i in range(len(lon_steps)):
        for j in range(len(lat_steps)):
            tile_min_lon = lon_steps[i]
            tile_min_lat = lat_steps[j]
            tile_max_lon = min(lon_steps[i] + tile_size_degrees, max_lon)
            tile_max_lat = min(lat_steps[j] + tile_size_degrees, max_lat)
            if tile_max_lon > tile_min_lon and tile_max_lat > tile_min_lat:
                tile_bboxes.append((tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat))

    print(f"Divided overall bounds into {len(tile_bboxes)} tiles.")
    return tile_bboxes

# --- Main Script ---
if __name__ == "__main__":

    # --- Create Output Directory ---
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            print(f"Created output directory: {OUTPUT_DIR}")
        except OSError as e:
            print(f"Error creating output directory {OUTPUT_DIR}: {e}")
            exit()
    # ---

    # --- Option to Regenerate Map from Existing GeoJSON ---
    regenerate_map_only = False
    if os.path.exists(OUTPUT_GEOJSON_FILENAME):
        while True:
            regen_choice = input(f"Found existing '{OUTPUT_GEOJSON_FILENAME}'. Regenerate map only? (y/n): ").lower()
            if regen_choice in ['y', 'n']:
                if regen_choice == 'y':
                    regenerate_map_only = True
                break
            print("Invalid input. Please enter 'y' or 'n'.")

    if regenerate_map_only:
        print(f"\nRegenerating map using existing '{OUTPUT_GEOJSON_FILENAME}'...")
        try:
            print(f"Loading unmatched OSM features from {OUTPUT_GEOJSON_FILENAME}...")
            unmatched_osm_buildings = gpd.read_file(OUTPUT_GEOJSON_FILENAME)
            unmatched_osm_buildings = unmatched_osm_buildings.to_crs("EPSG:4326") # Ensure correct CRS
            print(f"Loaded {len(unmatched_osm_buildings)} features.")

            # --- Load Input Shapefiles for Context ---
            print("Loading input shapefiles for map context...")
            # Load Location Shapefile (gdf_shp_proj, gdf_shp_4326)
            if not os.path.exists(SHAPEFILE_PATH): raise FileNotFoundError(f"Location Shapefile not found: {SHAPEFILE_PATH}")
            gdf_shp_all = gpd.read_file(SHAPEFILE_PATH)
            gdf_shp_all.geometry = gdf_shp_all.geometry.scale(xfact=0.01, yfact=0.01, zfact=1.0, origin=(0, 0))
            corrected_crs = "EPSG:26917" # Assuming this is correct
            gdf_shp_all.crs = corrected_crs
            gdf_shp = gdf_shp_all[gdf_shp_all.geometry.is_valid & ~gdf_shp_all.geometry.is_empty].copy()
            gdf_shp = gdf_shp[gdf_shp.geometry.type.isin(['Point', 'Polygon', 'MultiPolygon'])]
            if gdf_shp.empty: raise ValueError("No valid features in location shapefile.")
            target_crs = gdf_shp.crs
            gdf_shp_proj = gdf_shp.to_crs(target_crs) # Keep projected version if needed later
            gdf_shp_4326 = gdf_shp.to_crs("EPSG:4326")

            # Load Mask Shapefile (gdf_mask_proj, gdf_mask_4326)
            if not os.path.exists(MASK_SHAPEFILE_PATH): raise FileNotFoundError(f"Mask Shapefile not found: {MASK_SHAPEFILE_PATH}")
            gdf_mask_all = gpd.read_file(MASK_SHAPEFILE_PATH)
            gdf_mask_all.geometry = gdf_mask_all.geometry.scale(xfact=0.01, yfact=0.01, zfact=1.0, origin=(0, 0))
            gdf_mask_all.crs = corrected_crs
            gdf_mask = gdf_mask_all[gdf_mask_all.geometry.is_valid & ~gdf_mask_all.geometry.is_empty].copy()
            gdf_mask = gdf_mask[gdf_mask.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            if gdf_mask.empty: raise ValueError("No valid polygons in mask shapefile.")
            gdf_mask_proj = gdf_mask.to_crs(target_crs) # Keep projected version if needed later
            gdf_mask_4326 = gdf_mask.to_crs("EPSG:4326")
            print("Input shapefiles loaded.")

            #Load Found Shapefile Masks

            if not os.path.exists(FOUND_SHAPEFILE_PATH): raise FileNotFoundError(f"Found Shapefile not found: {FOUND_SHAPEFILE_PATH}")

            gdf_found_all = gpd.read_file(FOUND_SHAPEFILE_PATH)
            gdf_found_all.geometry = gdf_found_all.geometry.scale(xfact=0.01, yfact=0.01, zfact=1.0, origin=(0, 0))
            gdf_found_all.crs = corrected_crs
            gdf_found = gdf_found_all[gdf_found_all.geometry.is_valid & ~gdf_found_all.geometry.is_empty].copy()
            gdf_found = gdf_found[gdf_found.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            if gdf_found.empty: raise ValueError("No valid features in found shapefile.")
            gdf_found_proj = gdf_found.to_crs(target_crs) # Keep projected version if needed later
            gdf_found_4326 = gdf_found.to_crs("EPSG:4326")
            print("Found shapefile loaded.")


            # --- End Loading Input Shapefiles ---

            # --- Initialize and Populate Map ---
            print("Initializing map...")
            map_center_lat = gdf_shp_4326.geometry.union_all().centroid.y
            map_center_lon = gdf_shp_4326.geometry.union_all().centroid.x
            m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=10, tiles="OpenStreetMap")

            print("Adding layers to map...")
            # Add Input Masks (Blue)
            if not gdf_mask_4326.empty:
                folium.GeoJson(
                    gdf_mask_4326,
                    name='Input Masks (Blue)',
                    style_function=lambda x: {'color': 'blue', 'fillColor': 'blue', 'weight': 1, 'fillOpacity': 0.4}
                ).add_to(m)
            # Add Previously Found Masks (Brown)
            if not gdf_found_4326.empty:
                folium.GeoJson(
                    gdf_found_4326,
                    name = 'Found Masks (Brown)',
                    style_function=lambda x: {'color': 'brown', 'fillColor': 'brown', 'weight': 1, 'fillOpacity': 0.4},
                ).add_to(m)

            # Add Input Locations (Red Circles) - Corrected Tooltip & Timestamp Handling
            if not gdf_shp_4326.empty:
                gdf_shp_4326_map = gdf_shp_4326.copy() # Work on a copy
                for col in gdf_shp_4326_map.select_dtypes(include=['datetime64[ns]']).columns:
                    print(f"   Converting timestamp column '{col}' to string for map.")
                    gdf_shp_4326_map[col] = gdf_shp_4326_map[col].astype(str)

                tooltip_field = 'id' if 'id' in gdf_shp_4326_map.columns else (gdf_shp_4326_map.index.name if gdf_shp_4326_map.index.name else 'index')
                tooltip_alias = 'Location ID:' if tooltip_field == 'id' else 'Location Index:'
                if tooltip_field not in gdf_shp_4326_map.columns and tooltip_field == 'index':
                     gdf_shp_4326_map = gdf_shp_4326_map.reset_index() # Add index as column if needed

                folium.GeoJson(
                    gdf_shp_4326_map,
                    name='Input Locations (Red Circles)',
                    marker=folium.CircleMarker(radius=3, color='red', fillColor='red', fillOpacity=0.7),
                    tooltip=folium.features.GeoJsonTooltip(fields=[tooltip_field], aliases=[tooltip_alias]) # Use 'id' or fallback
                ).add_to(m)

            # Add Unmatched OSM Features (Green)
            if not unmatched_osm_buildings.empty:
                folium.GeoJson(
                    unmatched_osm_buildings,
                    name='Unmatched OSM Features (Green)',
                    style_function=lambda x: {'color': 'green', 'fillColor': 'green', 'weight': 1, 'fillOpacity': 0.6},
                    tooltip=folium.features.GeoJsonTooltip(fields=['osm_id'], aliases=['Unmatched OSM ID:'])
                ).add_to(m)
            else:
                print("Loaded GeoJSON contained no features.")

            folium.LayerControl().add_to(m)
            # --- End Map Population ---

            # --- Save Map ---
            print(f"Saving regenerated map to {OUTPUT_FILENAME}...")
            m.save(OUTPUT_FILENAME)
            print(f"Successfully saved regenerated map to {OUTPUT_FILENAME}")
            exit() # Exit after successful map regeneration

        except Exception as e:
            print(f"\nError during map regeneration: {e}")
            print(traceback.format_exc())
            print("Proceeding with full analysis instead.")
            # Fall through to full analysis if regeneration fails
    # --- End Map Regeneration Option ---

    # --- Full Analysis Starts Here (if not regenerating map) ---
    while True:
        mode = input("Enter debugging? (y/n)").lower()
        if mode in ['y', 'n']:
            break
        print("Invalid input. Please enter 'y' or 'n'")
    if mode == 'y':
        print("Debugging mode enabled.")
        debug = True
    else:
        print("Debugging mode disabled.")
        debug = False

    print(f"Starting processing for location shapefile: {SHAPEFILE_PATH}")
    if not os.path.exists(SHAPEFILE_PATH):
        print(f"Error: Location Shapefile not found at {SHAPEFILE_PATH}")
        exit()
    try:
        gdf_shp_all = gpd.read_file(SHAPEFILE_PATH)
        print(f"Read {len(gdf_shp_all)} features from location shapefile.")
        print(f"Original CRS reported by file: {gdf_shp_all.crs}")

        print("Scaling geometry coordinates from Centimeters to Meters (dividing by 100)...")
        gdf_shp_all.geometry = gdf_shp_all.geometry.scale(xfact=0.01, yfact=0.01, zfact=1.0, origin=(0, 0))

        corrected_crs = "EPSG:26917"
        print(f"Assigning correct meter-based CRS: {corrected_crs}")
        gdf_shp_all.crs = corrected_crs

        if SHAPEFILE_FILTER_COLUMN and SHAPEFILE_FILTER_VALUE:
            if SHAPEFILE_FILTER_COLUMN in gdf_shp_all.columns:
                gdf_shp = gdf_shp_all[gdf_shp_all[SHAPEFILE_FILTER_COLUMN] == SHAPEFILE_FILTER_VALUE].copy()
                print(f"Filtered shapefile to {len(gdf_shp)} features where '{SHAPEFILE_FILTER_COLUMN}' is '{SHAPEFILE_FILTER_VALUE}'.")
                if gdf_shp.empty:
                    print("Error: Filtering resulted in zero features. Check filter column/value.")
                    exit()
            else:
                print(f"Warning: Filter column '{SHAPEFILE_FILTER_COLUMN}' not found in shapefile. Using all features.")
                gdf_shp = gdf_shp_all.copy()
        else:
            gdf_shp = gdf_shp_all.copy()

        original_count = len(gdf_shp)
        gdf_shp = gdf_shp[gdf_shp.geometry.is_valid & ~gdf_shp.geometry.is_empty]
        gdf_shp = gdf_shp[gdf_shp.geometry.type.isin(['Point', 'Polygon', 'MultiPolygon'])]
        if len(gdf_shp) < original_count:
            print(f"Removed {original_count - len(gdf_shp)} invalid/empty/non-point/polygon geometries from shapefile data.")

        if gdf_shp.empty:
            print("Error: No valid features found in the location shapefile after filtering/validation.")
            exit()

        print("\nProjecting location shapefile data to EPSG:4326 for tile intersection...")
        try:
            gdf_shp_4326 = gdf_shp.to_crs("EPSG:4326")
            print("Projection to EPSG:4326 successful.")
        except Exception as e:
            print(f"ERROR: Could not project location shapefile to EPSG:4326: {e}")
            exit()

        target_crs = gdf_shp.crs
        gdf_shp_proj = gdf_shp

    except Exception as e:
        print(f"Error reading, scaling, or processing location shapefile: {e}")
        exit()

    print(f"\nLoading building mask shapefile: {MASK_SHAPEFILE_PATH}")
    if not os.path.exists(MASK_SHAPEFILE_PATH):
        print(f"Error: Mask Shapefile not found at {MASK_SHAPEFILE_PATH}")
        exit()
    try:
        gdf_mask_all = gpd.read_file(MASK_SHAPEFILE_PATH)
        print(f"Read {len(gdf_mask_all)} features from mask shapefile.")
        print(f"Original CRS reported by mask file: {gdf_mask_all.crs}")

        print("Scaling mask geometry coordinates from Centimeters to Meters (dividing by 100)...")
        gdf_mask_all.geometry = gdf_mask_all.geometry.scale(xfact=0.01, yfact=0.01, zfact=1.0, origin=(0, 0))

        print(f"Assigning correct meter-based CRS: {corrected_crs}")
        gdf_mask_all.crs = corrected_crs

        original_mask_count = len(gdf_mask_all)
        gdf_mask = gdf_mask_all[gdf_mask_all.geometry.is_valid & ~gdf_mask_all.geometry.is_empty].copy()
        gdf_mask = gdf_mask[gdf_mask.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if len(gdf_mask) < original_mask_count:
            print(f"Removed {original_mask_count - len(gdf_mask)} invalid/empty/non-polygon geometries from mask shapefile data.")

        if gdf_mask.empty:
            print("Error: No valid Polygon/MultiPolygon geometries found in the mask shapefile after validation.")
            exit()

        print(f"Projecting mask shapefile data to target CRS: {target_crs}...")
        gdf_mask_proj = gdf_mask.to_crs(target_crs)
        print("Projection successful.")
        print(f"Mask GeoDataFrame ready for comparison with {len(gdf_mask_proj)} features.")

        print("Projecting mask shapefile data to EPSG:4326 for map layer...")
        gdf_mask_4326 = gdf_mask_proj.to_crs("EPSG:4326")
        print("Mask projection to EPSG:4326 successful.")

    except Exception as e:
        print(f"Error reading, scaling, or processing mask shapefile: {e}")
        exit()

    line_buffer_proj = None
    line_buffer_4326 = None
    if LINE_SHAPEFILE_PATH:
        print(f"\nApplying line proximity filter using: {LINE_SHAPEFILE_PATH}")
        if not os.path.exists(LINE_SHAPEFILE_PATH):
            print(f"Error: Line shapefile not found at {LINE_SHAPEFILE_PATH}")
            exit()
        try:
            gdf_line = gpd.read_file(LINE_SHAPEFILE_PATH)
            if gdf_line.empty:
                raise ValueError("Line shapefile is empty.")
            valid_line_types = ['LineString', 'MultiLineString']
            gdf_line = gdf_line[gdf_line.geometry.type.isin(valid_line_types)]
            if gdf_line.empty:
                raise ValueError(f"No geometries of type {valid_line_types} found in the line file.")

            print(f"Read {len(gdf_line)} line features.")
            gdf_line_proj = gdf_line.to_crs(target_crs)
            print(f"Buffering line(s) by {FILTER_DISTANCE_METERS} meters...")
            line_buffer_proj = gdf_line_proj.geometry.buffer(FILTER_DISTANCE_METERS).union_all()

            print("Reprojecting line buffer to EPSG:4326 for tile skipping...")
            line_buffer_4326 = gpd.GeoSeries([line_buffer_proj], crs=target_crs).to_crs("EPSG:4326").iloc[0]

            print("Filtering input buildings by proximity to line buffer...")
            original_count = len(gdf_shp_proj)
            gdf_shp_proj = gdf_shp_proj[gdf_shp_proj.geometry.intersects(line_buffer_proj)]
            print(f"Filtered input buildings from {original_count} to {len(gdf_shp_proj)}.")
            if gdf_shp_proj.empty:
                print("Error: No input buildings found within the specified distance of the line.")
                exit()
        except Exception as e:
            print(f"Error processing line shapefile or filtering: {e}")
            exit()
    else:
        print("\nSkipping line proximity filter (LINE_SHAPEFILE_PATH not set).")

    print("\nProjecting potentially filtered location shapefile data to EPSG:4326...")
    try:
        gdf_shp_4326 = gdf_shp_proj.to_crs("EPSG:4326")
        print("Projection to EPSG:4326 successful.")
    except Exception as e:
        print(f"ERROR: Could not project filtered location shapefile to EPSG:4326: {e}")
        exit()

    try:
        print(f"\nCalculating overall query bounds directly from filtered EPSG:4326 data with margin...")
        base_bounds_4326 = gdf_shp_4326.total_bounds
        print(f"Base Bounds (EPSG:4326): {base_bounds_4326}")
        shp_bounds_poly_4326 = box(*base_bounds_4326)

        if line_buffer_4326 is not None:
            print("Using line buffer bounds for tiling extent.")
            tiling_base_bounds = line_buffer_4326.bounds
        else:
            print("Using filtered building bounds for tiling extent.")
            tiling_base_bounds = base_bounds_4326

        buffer_margin_deg = BUFFER_AREA_METERS / 111000.0
        print(f"Adding margin of ~{buffer_margin_deg:.5f} degrees (~{BUFFER_AREA_METERS}m) to tiling extent")

        overall_query_bounds_4326 = (
            tiling_base_bounds[0] - buffer_margin_deg,
            tiling_base_bounds[1] - buffer_margin_deg,
            tiling_base_bounds[2] + buffer_margin_deg,
            tiling_base_bounds[3] + buffer_margin_deg
        )
        print(f"Calculated overall query bounds for tiling (EPSG:4326): {overall_query_bounds_4326}")
    except Exception as e:
        print(f"Error calculating overall query bounds: {e}")
        exit()

    print("\nGenerating and filtering tile bounding boxes...")
    tile_bboxes = create_tile_bboxes(overall_query_bounds_4326, TILE_SIZE_DEGREES)
    initial_tile_count = len(tile_bboxes)

    if line_buffer_4326 is not None:
        print(f"Filtering {initial_tile_count} tiles by intersection with line buffer...")
        filtered_tile_bboxes = [
            bbox for bbox in tile_bboxes if box(*bbox).intersects(line_buffer_4326)
        ]
        tile_bboxes = filtered_tile_bboxes
        print(f"Kept {len(tile_bboxes)} tiles intersecting the line buffer area.")
    else:
        print("No line buffer provided, using all generated tiles.")

    total_tiles = len(tile_bboxes)
    if total_tiles == 0:
        print("Error: No tiles intersect the required area. Exiting.")
        exit()

    print("\nInitializing overall map...")
    map_center_lat = shp_bounds_poly_4326.centroid.y
    map_center_lon = shp_bounds_poly_4326.centroid.x
    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=10, tiles="OpenStreetMap")

    all_osm_gdfs = []
    all_shp_tiles_simple = []
    queried_tile_polys = []
    skipped_tiles = 0

    if debug:
        print(f"Debugging: Only checking some tiles")
        while True:
            num_tiles_str = input(f"How many tiles do you want to check (max {total_tiles})?\n").lower()
            if num_tiles_str.isdigit():
                num_tiles = int(num_tiles_str)
                if 0 < num_tiles <= total_tiles:
                    break
                else:
                    print(f"Please enter a number between 1 and {total_tiles}.")
            else:
                print("Invalid input. Please enter an integer.")

        print(f"Processing the first {num_tiles} tiles...")

        for i, tile_bbox in enumerate(tile_bboxes[:num_tiles]):
            tile_num = i + 1
            print(f"\n--- Checking Tile {tile_num}/{num_tiles} (Bounds: {tile_bbox}) ---")

            tile_poly_4326 = box(*tile_bbox)
            queried_tile_polys.append(tile_poly_4326)
            gdf_tile_osm = get_osm_buildings_gdf(tile_bbox, OVERPASS_API_URL, OVERPASS_TIMEOUT_SECONDS, USER_AGENT)

            osm_data_exists = False
            if gdf_tile_osm is not None and not gdf_tile_osm.empty:
                print(f"Debug Tile {tile_num}: Found {len(gdf_tile_osm)} OSM features.")
                all_osm_gdfs.append(gdf_tile_osm)
                osm_data_exists = True

            try:
                gdf_shp_tile_4326 = gdf_shp_4326[gdf_shp_4326.geometry.intersects(tile_poly_4326)]
                print(f"Debug Tile {tile_num}: Found {len(gdf_shp_tile_4326)} shapefile features intersecting EPSG:4326 tile bounds.")

                if not gdf_shp_tile_4326.empty:
                    gdf_shp_tile_4326['orig_index'] = gdf_shp_tile_4326.index.astype(str)
                    gdf_shp_tile_simple = gdf_shp_tile_4326[['geometry', 'orig_index']].copy()
                    all_shp_tiles_simple.append(gdf_shp_tile_simple)
            except Exception as intersect_err:
                print(f"Error during EPSG:4326 'intersects' check for tile {tile_num}: {intersect_err}")

            if i < num_tiles - 1:
                print(f"Waiting {TILE_REQUEST_DELAY_SECONDS}s before next request...")
                time.sleep(TILE_REQUEST_DELAY_SECONDS)

    if not debug:
        print(f"\nStarting Overpass queries for {total_tiles} tiles...")
        for i, tile_bbox in enumerate(tile_bboxes):
            tile_num = i + 1
            print(f"\n--- Checking Tile {tile_num}/{total_tiles} (Bounds: {tile_bbox}) ---")

            tile_poly_4326 = box(*tile_bbox)
            queried_tile_polys.append(tile_poly_4326)
            gdf_tile_osm = get_osm_buildings_gdf(tile_bbox, OVERPASS_API_URL, OVERPASS_TIMEOUT_SECONDS, USER_AGENT)

            osm_data_exists = False
            if gdf_tile_osm is not None and not gdf_tile_osm.empty:
                print(f"Tile {tile_num}: Found {len(gdf_tile_osm)} OSM features.")
                all_osm_gdfs.append(gdf_tile_osm)
                osm_data_exists = True

            try:
                gdf_shp_tile_4326 = gdf_shp_4326[gdf_shp_4326.geometry.intersects(tile_poly_4326)]
                print(f"Tile {tile_num}: Found {len(gdf_shp_tile_4326)} shapefile features intersecting EPSG:4326 tile bounds.")

                if not gdf_shp_tile_4326.empty:
                    gdf_shp_tile_4326['orig_index'] = gdf_shp_tile_4326.index.astype(str)
                    gdf_shp_tile_simple = gdf_shp_tile_4326[['geometry', 'orig_index']].copy()
                    all_shp_tiles_simple.append(gdf_shp_tile_simple)
            except Exception as intersect_err:
                print(f"Error during EPSG:4326 'intersects' check for tile {tile_num}: {intersect_err}")

            if i < total_tiles - 1:
                print(f"Waiting {TILE_REQUEST_DELAY_SECONDS}s before next request...")
                time.sleep(TILE_REQUEST_DELAY_SECONDS)

    print(f"\nFinished processing tiles. Skipped {skipped_tiles} tiles that did not intersect the core shapefile area.")

    print("\nCombining results for the final map...")

    gdf_tile_boundaries = None
    if queried_tile_polys:
        try:
            gdf_tile_boundaries = gpd.GeoDataFrame(geometry=queried_tile_polys, crs="EPSG:4326")
            print(f"Created GeoDataFrame for {len(gdf_tile_boundaries)} queried tile boundaries.")
        except Exception as e:
            print(f"Error creating tile boundaries GeoDataFrame: {e}")

    gdf_osm = None
    if not all_osm_gdfs:
        print("Warning: No OSM data was retrieved from any queried tile.")
    else:
        try:
            gdf_osm_combined = pd.concat(all_osm_gdfs, ignore_index=True)
            print(f"Combined OSM features before deduplication: {len(gdf_osm_combined)}")
            gdf_osm = gdf_osm_combined.drop_duplicates(subset=['osm_id'], keep='first')
            gdf_osm = gpd.GeoDataFrame(gdf_osm, geometry='geometry', crs="EPSG:4326")
            print(f"Combined OSM features after deduplication: {len(gdf_osm)}")

            if line_buffer_proj is not None:
                print("Filtering combined OSM data by proximity to line buffer...")
                try:
                    line_buffer_4326 = gpd.GeoSeries([line_buffer_proj], crs=target_crs).to_crs("EPSG:4326").iloc[0]
                    original_osm_count = len(gdf_osm)
                    gdf_osm = gdf_osm[gdf_osm.geometry.intersects(line_buffer_4326)]
                    print(f"Filtered combined OSM buildings from {original_osm_count} to {len(gdf_osm)}.")
                except Exception as e:
                    print(f"Warning: Could not filter OSM data by line proximity: {e}")
        except Exception as e:
            print(f"Error combining or deduplicating OSM tile results: {e}")
            gdf_osm = None

    gdf_shp_queried_simple = None
    if not all_shp_tiles_simple:
        print("Warning: No shapefile features found within any queried tile.")
    else:
        try:
            gdf_shp_queried_simple = pd.concat(all_shp_tiles_simple, ignore_index=True)
            gdf_shp_queried_simple = gdf_shp_queried_simple.drop_duplicates(subset=['orig_index'], keep='first')
            gdf_shp_queried_simple = gpd.GeoDataFrame(gdf_shp_queried_simple, geometry='geometry', crs="EPSG:4326")
            print(f"Combined {len(gdf_shp_queried_simple)} unique shapefile features found in queried tiles.")
        except Exception as e:
            print(f"Error combining shapefile tile results: {e}")
            gdf_shp_queried_simple = None

    print("\nAdding combined layers to the overall map...")

    if gdf_tile_boundaries is not None and not gdf_tile_boundaries.empty:
        folium.GeoJson(
            gdf_tile_boundaries,
            name='Queried Tile Boundaries (Red)',
            style_function=lambda x: {'color': 'red', 'fill': False, 'weight': 1}
        ).add_to(m)

    try:
        if not gdf_mask_4326.empty:
            folium.GeoJson(
                gdf_mask_4326,
                name='Input Masks (Blue)',
                style_function=lambda x: {'color': 'blue', 'fillColor': 'blue', 'weight': 1, 'fillOpacity': 0.4}
            ).add_to(m)
    except Exception as e:
        print(f"Warning: Could not add mask layer to map: {e}")

    try:
        if not gdf_shp_4326.empty:
            temp_gdf_shp_4326 = gdf_shp_4326.copy() # Work on a copy
            for col in temp_gdf_shp_4326.select_dtypes(include=['datetime64[ns]']).columns:
                 print(f"   Converting timestamp column '{col}' to string for map.")
                 temp_gdf_shp_4326[col] = temp_gdf_shp_4326[col].astype(str)

            tooltip_field = 'id' if 'id' in temp_gdf_shp_4326.columns else (temp_gdf_shp_4326.index.name if temp_gdf_shp_4326.index.name else 'index')
            tooltip_alias = 'Location ID:' if tooltip_field == 'id' else 'Location Index:'
            if tooltip_field not in temp_gdf_shp_4326.columns:
                 if tooltip_field == 'index' and temp_gdf_shp_4326.index.name is None:
                      temp_gdf_shp_4326.index.name = 'index'
                 temp_gdf_shp_4326 = temp_gdf_shp_4326.reset_index()
                 if tooltip_field not in temp_gdf_shp_4326.columns:
                      tooltip_field = temp_gdf_shp_4326.columns[0]
                      tooltip_alias = f"{tooltip_field}:"
                      print(f"Warning: Tooltip field for locations defaulting to '{tooltip_field}'.")

            folium.GeoJson(
                temp_gdf_shp_4326,
                name='Input Locations (Red Circles)',
                marker=folium.CircleMarker(radius=3, color='red', fillColor='red', fillOpacity=0.7),
                tooltip=folium.features.GeoJsonTooltip(fields=[tooltip_field], aliases=[tooltip_alias])
            ).add_to(m)
    except Exception as e:
        print(f"Warning: Could not add location points layer to map: {e}")

    unmatched_osm_buildings = None
    matched_osm_buildings = None
    matched_osm_indices = pd.Index([])

    if gdf_osm is not None and not gdf_osm.empty:
        matched_by_intersection_indices = pd.Index([])
        try:
            gdf_osm_proj = gdf_osm.to_crs(target_crs)
            gdf_osm_proj_polygons = gdf_osm_proj[gdf_osm_proj.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            print(f"\nCombined OSM data reprojected to {target_crs} ({len(gdf_osm_proj)} features, {len(gdf_osm_proj_polygons)} polygons).")

            gdf_mask_proj_polygons = gdf_mask_proj[gdf_mask_proj.geometry.type.isin(['Polygon', 'MultiPolygon'])]

            if not gdf_osm_proj_polygons.empty and not gdf_mask_proj_polygons.empty:
                print(f"\n1. Comparing mask polygons with OSM polygons using intersection...")
                if gdf_mask_proj_polygons.geometry.name != 'geometry':
                    gdf_mask_proj_polygons = gdf_mask_proj_polygons.rename_geometry('geometry')
                if gdf_osm_proj_polygons.geometry.name != 'geometry':
                    gdf_osm_proj_polygons = gdf_osm_proj_polygons.rename_geometry('geometry')

                intersected_gdf = gpd.sjoin(gdf_mask_proj_polygons, gdf_osm_proj_polygons,
                                            how='inner', predicate='intersects',
                                            lsuffix='mask', rsuffix='osm')
                print(f"   Found {len(intersected_gdf)} intersections.")

                if not intersected_gdf.empty:
                    matched_osm_indices_in_join = intersected_gdf['index_osm'].dropna().unique()
                    valid_indices = matched_osm_indices_in_join[np.isin(matched_osm_indices_in_join, gdf_osm.index)]
                    matched_by_intersection_indices = gdf_osm.loc[valid_indices].index
            else:
                print("   Skipping intersection: No OSM or Mask polygons available.")

        except Exception as e:
            print(f"   Error during intersection spatial join: {e}")

        matched_by_distance_indices = pd.Index([])
        all_osm_indices_proj = gdf_osm_proj.index
        potential_distance_match_indices = all_osm_indices_proj.difference(matched_by_intersection_indices)

        if not potential_distance_match_indices.empty:
            gdf_osm_proj_subset = gdf_osm_proj.loc[potential_distance_match_indices]
            print(f"\n2. Comparing {len(gdf_osm_proj_subset)} remaining OSM features with location shapefile using distance ({MATCHING_DISTANCE_METERS}m)...")

            try:
                if gdf_shp_proj.geometry.name != 'geometry':
                    gdf_shp_proj = gdf_shp_proj.rename_geometry('geometry')
                if gdf_osm_proj_subset.geometry.name != 'geometry':
                    gdf_osm_proj_subset = gdf_osm_proj_subset.rename_geometry('geometry')

                distance_joined_gdf = gpd.sjoin_nearest(gdf_shp_proj, gdf_osm_proj_subset,
                                                        how='inner',
                                                        max_distance=MATCHING_DISTANCE_METERS,
                                                        lsuffix='shp', rsuffix='osm')
                print(f"   Found {len(distance_joined_gdf)} matches within distance.")

                if not distance_joined_gdf.empty:
                    matched_osm_indices_in_dist_join = distance_joined_gdf['index_osm'].dropna().unique()
                    matched_by_distance_indices = pd.Index(matched_osm_indices_in_dist_join)

            except Exception as e:
                print(f"   Error during distance spatial join: {e}")
        else:
            print("\n2. No remaining OSM features to check by distance.")

        print("\n--- Processing Combined Comparison Results ---")
        matched_osm_indices = matched_by_intersection_indices.union(matched_by_distance_indices)

        if not matched_osm_indices.empty:
            matched_osm_buildings = gdf_osm.loc[matched_osm_indices]

        all_original_osm_indices = gdf_osm.index
        unmatched_osm_indices = all_original_osm_indices.difference(matched_osm_indices)
        unmatched_osm_buildings = gdf_osm.loc[unmatched_osm_indices]

        print(f"   Matched by Intersection: {len(matched_by_intersection_indices)}")
        print(f"   Matched by Distance (only): {len(matched_by_distance_indices)}")
        print(f"   Total Matched (Combined): {len(matched_osm_indices)}")
        print(f"   Total Unmatched: {len(unmatched_osm_indices)}")

    if matched_osm_buildings is not None and not matched_osm_buildings.empty:
        folium.GeoJson(
            matched_osm_buildings,
            name='Matched OSM Features (Cyan)',
            style_function=lambda x: {'color': 'cyan', 'fillColor': 'cyan', 'weight': 1, 'fillOpacity': 0.7},
            tooltip=folium.features.GeoJsonTooltip(fields=['osm_id'], aliases=['Matched OSM ID:'])
        ).add_to(m)

    if unmatched_osm_buildings is not None and not unmatched_osm_buildings.empty:
        folium.GeoJson(
            unmatched_osm_buildings,
            name='Unmatched OSM Features (Green)',
            style_function=lambda x: {'color': 'green', 'fillColor': 'green', 'weight': 1, 'fillOpacity': 0.6},
            tooltip=folium.features.GeoJsonTooltip(fields=['osm_id'], aliases=['Unmatched OSM ID:'])
        ).add_to(m)
    elif gdf_osm is not None:
         if not matched_osm_indices.empty:
              print("No unmatched OSM features found to add to map.")

    folium.LayerControl().add_to(m)
    try:
        print(f"Saving overall map to {OUTPUT_FILENAME}...")
        m.save(OUTPUT_FILENAME)
        print(f"Successfully saved overall map to {OUTPUT_FILENAME}")
    except Exception as e:
        print(f"ERROR: Could not save overall map: {e}")
        print(traceback.format_exc())

    if unmatched_osm_buildings is not None:
        if not unmatched_osm_buildings.empty:
            try:
                print(f"Saving {len(unmatched_osm_buildings)} unmatched OSM features to {OUTPUT_GEOJSON_FILENAME}...")
                if unmatched_osm_buildings.geometry.name != 'geometry':
                    unmatched_osm_buildings = unmatched_osm_buildings.rename_geometry('geometry')
                unmatched_osm_buildings.to_file(OUTPUT_GEOJSON_FILENAME, driver='GeoJSON')
                print("Save successful.")
            except Exception as e:
                print(f"Error saving unmatched OSM features: {e}")
                print(traceback.format_exc())
        else:
            print("No unmatched OSM features found to save.")
    elif gdf_osm is None:
        print("\nError: Cannot save output - No combined OSM data available or comparison failed.")

    print("\nProcessing finished.")