import json
import os
import shutil
import glob
import re
import pickle
from pathlib import Path
from collections import defaultdict
from shapely.ops import transform
import orbital_vault as ov
from pimsys.regions.RegionsDb import RegionsDb
import rasterio
from shapely.geometry import box, shape
import pyproj

# Cache directory for persistent storage
CACHE_DIR = "/cephfs/work/rithvik/datasets/datasets/GlobalBuildings/.cache/"
os.makedirs(CACHE_DIR, exist_ok=True)

# Coordinate system transformers
# Valid areas are in WGS84 (EPSG:4326), buildings are in Web Mercator (EPSG:3857)
transformer_4326_to_3857 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
transformer_3857_to_4326 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

def parse_geojson_filename(filename):
    """
    Parse GeoJSON filename to extract coverage bounds.
    Example: w080_n35_w075_n30.geojson -> (west1=-80, north1=35, west2=-75, north2=30)
    Returns: (min_lon, min_lat, max_lon, max_lat)
    """
    basename = os.path.basename(filename)
    # Pattern: w{west1}_n{north1}_w{west2}_n{north2}.geojson
    match = re.match(r'w(\d+)_n(\d+)_w(\d+)_n(\d+)\.geojson', basename)
    if match:
        w1, n1, w2, n2 = map(int, match.groups())
        # Convert to negative for western hemisphere
        min_lon = -w1
        max_lon = -w2
        min_lat = n2
        max_lat = n1
        return (min_lon, min_lat, max_lon, max_lat)
    return None

def find_relevant_geojson_files(valid_areas_bounds, geojson_dir):
    """
    Find all GeoJSON files that overlap with the given bounds.
    valid_areas_bounds: (min_lon, min_lat, max_lon, max_lat)
    """
    geojson_files = glob.glob(os.path.join(geojson_dir, "*.geojson"))
    relevant_files = []
    
    va_min_lon, va_min_lat, va_max_lon, va_max_lat = valid_areas_bounds
    
    for geojson_file in geojson_files:
        bounds = parse_geojson_filename(geojson_file)
        if bounds:
            gj_min_lon, gj_min_lat, gj_max_lon, gj_max_lat = bounds
            
            # Check if bounding boxes overlap
            if not (va_max_lon < gj_min_lon or va_min_lon > gj_max_lon or
                    va_max_lat < gj_min_lat or va_min_lat > gj_max_lat):
                relevant_files.append(geojson_file)
    
    return relevant_files

def get_cache_path(geojson_file):
    """Get the cache file path for a GeoJSON file (spatial index)."""
    basename = os.path.basename(geojson_file)
    cache_filename = basename.replace('.geojson', '_spatial_index.pkl')
    return os.path.join(CACHE_DIR, cache_filename)

def is_cache_valid(geojson_file, cache_path):
    """Check if cache file exists and is newer than the GeoJSON file."""
    if not os.path.exists(cache_path):
        return False
    
    geojson_mtime = os.path.getmtime(geojson_file)
    cache_mtime = os.path.getmtime(cache_path)
    
    return cache_mtime > geojson_mtime

def create_spatial_index(geojson_file, grid_size_meters=10000):
    """
    Create a spatial grid index for fast spatial queries.
    Buildings are grouped by grid cells for O(1) spatial lookup.
    grid_size_meters: size of each grid cell in meters (EPSG:3857 units)
    """
    print(f"  Creating spatial index for: {os.path.basename(geojson_file)}")
    print(f"    Grid size: {grid_size_meters}m")
    
    try:
        with open(geojson_file, 'r') as f:
            data = json.load(f)
        
        total_features = len(data['features'])
        print(f"    Total buildings: {total_features}")
        
        # Create grid index: grid_cell -> list of (building_geom_wgs84, building_index)
        spatial_index = defaultdict(list)
        
        # Create a transform function for shapely
        from shapely.ops import transform as shapely_transform
        
        for i, feature in enumerate(data['features']):
            if i > 0 and i % 100000 == 0:
                print(f"    Indexing: {i}/{total_features} ({100*i//total_features}%)")
            
            building_geom_3857 = shape(feature['geometry'])  # In EPSG:3857
            
            # Transform to WGS84 for storage
            building_geom_wgs84 = shapely_transform(transformer_3857_to_4326.transform, building_geom_3857)
            
            # Get building bounds in EPSG:3857 for grid assignment
            minx, miny, maxx, maxy = building_geom_3857.bounds
            
            # Assign to grid cells (a building can be in multiple cells)
            min_grid_x = int(minx // grid_size_meters)
            max_grid_x = int(maxx // grid_size_meters)
            min_grid_y = int(miny // grid_size_meters)
            max_grid_y = int(maxy // grid_size_meters)
            
            for gx in range(min_grid_x, max_grid_x + 1):
                for gy in range(min_grid_y, max_grid_y + 1):
                    spatial_index[(gx, gy)].append(building_geom_wgs84)
        
        print(f"    ✓ Created spatial index with {len(spatial_index)} grid cells")
        return spatial_index, grid_size_meters
        
    except Exception as e:
        print(f"    ✗ Error creating spatial index: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def query_spatial_index(spatial_index, grid_size_meters, bounds_wgs84):
    """
    Query the spatial index to get buildings in the given bounds.
    Much faster than scanning all buildings.
    """
    # Transform bounds to EPSG:3857 to query grid
    min_x_3857, min_y_3857 = transformer_4326_to_3857.transform(bounds_wgs84[0], bounds_wgs84[1])
    max_x_3857, max_y_3857 = transformer_4326_to_3857.transform(bounds_wgs84[2], bounds_wgs84[3])
    
    # Find relevant grid cells
    min_grid_x = int(min_x_3857 // grid_size_meters)
    max_grid_x = int(max_x_3857 // grid_size_meters)
    min_grid_y = int(min_y_3857 // grid_size_meters)
    max_grid_y = int(max_y_3857 // grid_size_meters)
    
    print(f"  Querying spatial index...")
    print(f"    Grid cells to check: {(max_grid_x - min_grid_x + 1) * (max_grid_y - min_grid_y + 1)}")
    
    # Collect buildings from relevant grid cells
    buildings = []
    for gx in range(min_grid_x, max_grid_x + 1):
        for gy in range(min_grid_y, max_grid_y + 1):
            if (gx, gy) in spatial_index:
                buildings.extend(spatial_index[(gx, gy)])
    
    print(f"    ✓ Found {len(buildings)} candidate buildings from index")
    return buildings

def load_buildings_for_bounds(geojson_file, bounds_wgs84):
    """
    Load only buildings that intersect with the given bounds.
    Uses spatial index for fast lookup - creates index on first use.
    bounds_wgs84: (minx, miny, maxx, maxy) in WGS84 (EPSG:4326)
    """
    cache_path = get_cache_path(geojson_file)
    
    # Try to load spatial index from cache
    if is_cache_valid(geojson_file, cache_path):
        print(f"  Loading spatial index from cache...")
        try:
            with open(cache_path, 'rb') as f:
                spatial_index, grid_size_meters = pickle.load(f)
            print(f"    ✓ Loaded spatial index with {len(spatial_index)} grid cells")
        except Exception as e:
            print(f"    ⚠ Cache load failed: {e}, creating new index...")
            spatial_index = None
    else:
        spatial_index = None
    
    # Create spatial index if not cached
    if spatial_index is None:
        spatial_index, grid_size_meters = create_spatial_index(geojson_file)
        if spatial_index:
            # Save to cache
            print(f"    Saving spatial index to cache...")
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump((spatial_index, grid_size_meters), f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"    ✓ Spatial index cached successfully")
            except Exception as e:
                print(f"    ⚠ Failed to save cache: {e}")
    
    # Query the spatial index
    if spatial_index:
        buildings = query_spatial_index(spatial_index, grid_size_meters, bounds_wgs84)
        return buildings
    else:
        print(f"    ⚠ Spatial index creation failed, falling back to linear scan")
        return []

def load_buildings_for_bounds_OLD(geojson_file, bounds_wgs84):
    """
    OLD VERSION: Load only buildings that intersect with the given bounds directly from GeoJSON.
    Much more efficient than loading everything.
    bounds_wgs84: (minx, miny, maxx, maxy) in WGS84 (EPSG:4326)
    
    Note: Building GeoJSON is in EPSG:3857 (Web Mercator), so we transform bounds
    """
    from shapely.ops import transform as shapely_transform
    
    # Transform bounds from WGS84 to Web Mercator to match building coordinates
    min_x_3857, min_y_3857 = transformer_4326_to_3857.transform(bounds_wgs84[0], bounds_wgs84[1])
    max_x_3857, max_y_3857 = transformer_4326_to_3857.transform(bounds_wgs84[2], bounds_wgs84[3])
    bounds_3857 = (min_x_3857, min_y_3857, max_x_3857, max_y_3857)
    bounds_box = box(*bounds_3857)
    
    print(f"  Loading buildings from GeoJSON for specific bounds...")
    print(f"    Bounds (WGS84): {bounds_wgs84}")
    print(f"    Bounds (EPSG:3857): ({bounds_3857[0]:.0f}, {bounds_3857[1]:.0f}, {bounds_3857[2]:.0f}, {bounds_3857[3]:.0f})")
    
    try:
        with open(geojson_file, 'r') as f:
            data = json.load(f)
        
        total_features = len(data['features'])
        print(f"    Total buildings in file: {total_features}")
        
        # Only load buildings that intersect with bounds (buildings are in EPSG:3857)
        buildings = []
        skipped = 0
        for i, feature in enumerate(data['features']):
            if i > 0 and i % 100000 == 0:
                print(f"    Scanning: {i}/{total_features} ({100*i//total_features}%) - found {len(buildings)}, skipped {skipped}")
            
            building_geom = shape(feature['geometry'])  # Already in EPSG:3857
            
            # Quick bounds check (both in EPSG:3857 now)
            if not bounds_box.intersects(box(*building_geom.bounds)):
                skipped += 1
                continue
            
            # Transform building to WGS84 to match valid areas for later processing
            building_wgs84 = shapely_transform(transformer_3857_to_4326.transform, building_geom)
            buildings.append(building_wgs84)
        
        print(f"    ✓ Loaded {len(buildings)} buildings (skipped {skipped} outside bounds)")
        print(f"    Buildings transformed from EPSG:3857 to WGS84")
        return buildings
    except Exception as e:
        print(f"    ✗ Error loading {os.path.basename(geojson_file)}: {e}")
        return []

def load_geojson_file(geojson_file):
    """Load a GeoJSON file and return all building geometries. Uses persistent cache."""
    cache_path = get_cache_path(geojson_file)
    
    # Try to load from cache first
    if is_cache_valid(geojson_file, cache_path):
        print(f"  Loading from cache: {os.path.basename(cache_path)}")
        try:
            with open(cache_path, 'rb') as f:
                buildings = pickle.load(f)
            print(f"    ✓ Loaded {len(buildings)} buildings from cache")
            return buildings
        except Exception as e:
            print(f"    ⚠ Cache load failed: {e}, reloading from GeoJSON...")
    
    # Load from GeoJSON file
    print(f"  Loading GeoJSON: {os.path.basename(geojson_file)}")
    try:
        with open(geojson_file, 'r') as f:
            data = json.load(f)
        
        total_features = len(data['features'])
        print(f"    Total buildings in file: {total_features}")
        
        # Convert all features to shapely geometries
        buildings = []
        for i, feature in enumerate(data['features']):
            if i > 0 and i % 100000 == 0:
                print(f"    Loading: {i}/{total_features} ({100*i//total_features}%)")
            buildings.append(shape(feature['geometry']))
        
        print(f"    ✓ Loaded {len(buildings)} buildings")
        
        # Save to cache
        print(f"    Saving to cache: {os.path.basename(cache_path)}")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(buildings, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"    ✓ Cache saved successfully")
        except Exception as e:
            print(f"    ⚠ Failed to save cache: {e}")
        
        return buildings
    except Exception as e:
        print(f"    ✗ Error loading {os.path.basename(geojson_file)}: {e}")
        return []

def filter_buildings_by_valid_areas(buildings, valid_areas_geom):
    """
    Filter pre-loaded buildings that intersect with valid areas.
    Uses bounds check for speed optimization.
    """
    intersecting_buildings = []
    valid_bounds_box = box(*valid_areas_geom.bounds)
    
    print(f"  Filtering {len(buildings)} buildings for valid areas...")
    
    checked = 0
    skipped_by_bounds = 0
    
    for i, building_geom in enumerate(buildings):
        # Progress indicator for large building lists
        if i > 0 and i % 50000 == 0:
            print(f"    Progress: {i}/{len(buildings)} ({100*i//len(buildings)}%) - skipped {skipped_by_bounds} by bounds check")
        
        # Quick bounds check first (MUCH faster than intersection)
        if not valid_bounds_box.intersects(box(*building_geom.bounds)):
            skipped_by_bounds += 1
            continue
        
        checked += 1
        # Only do expensive intersection check if bounds overlap
        if valid_areas_geom.intersects(building_geom):
            intersecting_buildings.append(building_geom)
    
    print(f"    ✓ Found {len(intersecting_buildings)} intersecting buildings")
    print(f"    Efficiently skipped {skipped_by_bounds}/{len(buildings)} buildings using bounds check")
    return intersecting_buildings

def load_buildings_from_geojson(geojson_files, valid_areas_geom, geojson_cache):
    """
    Load building polygons from GeoJSON files that intersect with valid areas.
    Optimized to only load buildings within the valid areas bounds.
    """
    all_buildings = []
    valid_bounds = valid_areas_geom.bounds
    
    for geojson_file in geojson_files:
        # Create a cache key that includes the bounds (for region-specific caching)
        cache_key = f"{geojson_file}_{valid_bounds[0]:.3f}_{valid_bounds[1]:.3f}_{valid_bounds[2]:.3f}_{valid_bounds[3]:.3f}"
        
        # Check if already loaded in cache for this region
        if cache_key in geojson_cache:
            print(f"  Using cached data for: {os.path.basename(geojson_file)} (region-specific)")
            buildings = geojson_cache[cache_key]
        else:
            # Load only buildings within the bounds directly from GeoJSON
            print(f"  Loading buildings from: {os.path.basename(geojson_file)}")
            buildings = load_buildings_for_bounds(geojson_file, valid_bounds)
            geojson_cache[cache_key] = buildings
        
        # Filter buildings for this specific valid area
        filtered_buildings = filter_buildings_by_valid_areas(buildings, valid_areas_geom)
        all_buildings.extend(filtered_buildings)
    
    return all_buildings

def geo_to_pixel(geo_x, geo_y, transform):
    """Convert geographic coordinates to pixel coordinates"""
    from rasterio.transform import rowcol
    row, col = rowcol(transform, geo_x, geo_y)
    return col, row

def polygon_to_yolo_bbox(polygon, image_width, image_height, transform):
    """Convert polygon to YOLO format bounding box"""
    # Get the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds
    
    # Convert geographic bounds to pixel coordinates
    left_px, top_px = geo_to_pixel(minx, maxy, transform)
    right_px, bottom_px = geo_to_pixel(maxx, miny, transform)
    
    # Ensure coordinates are within image bounds
    left_px = max(0, min(left_px, image_width - 1))
    right_px = max(0, min(right_px, image_width - 1))
    top_px = max(0, min(top_px, image_height - 1))
    bottom_px = max(0, min(bottom_px, image_height - 1))
    
    # Calculate YOLO format (center_x, center_y, width, height) normalized [0,1]
    bbox_width = abs(right_px - left_px)
    bbox_height = abs(bottom_px - top_px)
    center_x = (left_px + right_px) / 2
    center_y = (top_px + bottom_px) / 2
    
    # Normalize to [0,1]
    center_x_norm = center_x / image_width
    center_y_norm = center_y / image_height
    width_norm = bbox_width / image_width
    height_norm = bbox_height / image_height
    
    return center_x_norm, center_y_norm, width_norm, height_norm

# Paths
geojson_dir = "/cephfs/work/rithvik/datasets/datasets/GlobalBuildings/"
# Use filtered dataset (only images with GeoJSON coverage)
input_valid_areas_path = "/cephfs/work/rithvik/datasets/datasets/BHE/Maxar_images_filtered/valid_areas/"
input_images_path = "/cephfs/work/rithvik/datasets/datasets/BHE/Maxar_images_filtered/images/"



# Check that the valid_areas and images directories contain the same file names
# Create output directory structure
base_output_path = "/cephfs/work/rithvik/datasets/datasets/BHE/Maxar_GBA_subset/"
images_output_path = os.path.join(base_output_path, "images")
labels_output_path = os.path.join(base_output_path, "labels")
# Create directories if they don't exist
os.makedirs(images_output_path, exist_ok=True)
os.makedirs(labels_output_path, exist_ok=True)

# Initialize GeoJSON cache (avoids reloading large files multiple times)
geojson_cache = {}
print("="*70)
print("PERSISTENT CACHING ENABLED")
print("Large GeoJSON files will be:")
print("  1. Loaded once per script run (in-memory cache)")
print("  2. Saved to disk cache for future runs")
print(f"  3. Cache directory: {CACHE_DIR}")
print("Subsequent runs will be MUCH faster!")
print("="*70)

# Statistics
total_processed = 0
total_with_labels = 0
total_skipped_no_geojson = 0
total_skipped_no_valid_areas = 0
total_skipped_already_processed = 0

# Process all filtered images (they all have GeoJSON coverage)
images_processed_count = 0

for image in os.listdir(input_images_path):
    if not image.endswith('.tif'):
        continue
        
    input_path = os.path.join(input_images_path, image)
    image_basename = os.path.splitext(image)[0]
    
    # Check if already processed (label file exists)
    label_file = os.path.join(labels_output_path, f"{image_basename}.txt")
    if os.path.exists(label_file):
        print(f"\n{'='*70}")
        print(f"Skipping (already processed): {image}")
        print(f"{'='*70}")
        images_processed_count += 1
        total_skipped_already_processed += 1
        continue
    
    print(f"\n{'='*70}")
    print(f"Processing: {image}")
    print(f"{'='*70}")
    
    # Load corresponding valid areas file
    valid_areas_file = os.path.join(input_valid_areas_path, f"{image_basename}.geojson")
    
    if not os.path.exists(valid_areas_file):
        print(f"  ⚠ No valid areas file found, skipping...")
        total_skipped_no_valid_areas += 1
        images_processed_count += 1
        continue
    
    # Read valid areas geometry
    try:
        with open(valid_areas_file, 'r') as f:
            valid_areas_geojson = json.load(f)
        
        if valid_areas_geojson['type'] == 'FeatureCollection':
            valid_areas_geom = shape(valid_areas_geojson['features'][0]['geometry'])
        else:
            valid_areas_geom = shape(valid_areas_geojson['geometry'])
        
        print(f"  Valid areas geometry: {valid_areas_geom.geom_type}")
    except Exception as e:
        print(f"  ✗ Error reading valid areas: {e}")
        total_skipped_no_valid_areas += 1
        images_processed_count += 1
        continue
    
    # Get valid areas bounds
    valid_bounds = valid_areas_geom.bounds  # (minx, miny, maxx, maxy)
    print(f"  Valid areas bounds: {valid_bounds}")
    
    # Find relevant GeoJSON files
    relevant_geojson_files = find_relevant_geojson_files(valid_bounds, geojson_dir)
    
    if not relevant_geojson_files:
        print(f"  ⚠ No GeoJSON coverage for this area, skipping...")
        total_skipped_no_geojson += 1
        images_processed_count += 1
        continue
    
    print(f"  Found {len(relevant_geojson_files)} relevant GeoJSON file(s):")
    for gj_file in relevant_geojson_files:
        print(f"    - {os.path.basename(gj_file)}")
    
    # Load buildings from relevant GeoJSON files (using cache)
    building_polygons = load_buildings_from_geojson(relevant_geojson_files, valid_areas_geom, geojson_cache)
    print(f"  Total buildings for this image: {len(building_polygons)}")
    
    # Get image properties for coordinate transformation
    with rasterio.open(input_path) as src:
        image_bounds = src.bounds
        image_crs = src.crs
        image_width = src.width
        image_height = src.height
        transform = src.transform
    
    # Clip buildings to valid areas
    clipped_polygons = []
    for poly in building_polygons:
        # Clip the polygon to the valid areas
        clipped_poly = poly.intersection(valid_areas_geom)
        if not clipped_poly.is_empty:
            clipped_polygons.append(clipped_poly)
    
    print(f"  Clipped polygons: {len(clipped_polygons)}")

    # Convert polygons to YOLO format
    if not clipped_polygons:
        print(f"  ⚠ No buildings found in valid areas, skipping...")
        images_processed_count += 1
        continue
    
    # Create output filenames for images and labels folders
    image_name = os.path.splitext(image)[0]
    
    # Copy image to images folder
    image_output_path = os.path.join(images_output_path, image)
    shutil.copy2(input_path, image_output_path)
    
    # Create label file in labels folder
    label_output_path = os.path.join(labels_output_path, f"{image_name}.txt")
    
    yolo_bboxes = []
    for poly in clipped_polygons:
        if poly.geom_type == 'Polygon':
            bbox = polygon_to_yolo_bbox(poly, image_width, image_height, transform)
            # Filter out small boxes: check if width AND height are both >= 10 pixels
            width_pixels = bbox[2] * image_width
            height_pixels = bbox[3] * image_height
            if width_pixels >= 10 and height_pixels >= 10:
                yolo_bboxes.append(bbox)
        elif poly.geom_type == 'MultiPolygon':
            for single_poly in poly.geoms:
                bbox = polygon_to_yolo_bbox(single_poly, image_width, image_height, transform)
                # Filter out small boxes: check if width AND height are both >= 10 pixels
                width_pixels = bbox[2] * image_width
                height_pixels = bbox[3] * image_height
                if width_pixels >= 10 and height_pixels >= 10:
                    yolo_bboxes.append(bbox)
    
    # Write YOLO format labels to file
    with open(label_output_path, 'w') as f:
        for bbox in yolo_bboxes:
            # YOLO format: class_id center_x center_y width height
            # Using class_id = 0 for buildings
            f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    print(f"  ✓ Saved image to: {os.path.join('images', image)}")
    print(f"  ✓ Saved {len(yolo_bboxes)} labels to: {os.path.join('labels', f'{image_name}.txt')}")
    print(f"  Filtered out {len(clipped_polygons) - len(yolo_bboxes)} small boxes (< 10px)")
    
    total_processed += 1
    total_with_labels += 1
    images_processed_count += 1

# Print final summary
print("\n" + "="*70)
print("PROCESSING COMPLETE - SUMMARY")
print("="*70)
print(f"Total images processed successfully: {total_with_labels}")
print(f"Images skipped (already processed): {total_skipped_already_processed}")
print(f"Images skipped (no GeoJSON coverage): {total_skipped_no_geojson}")
print(f"Images skipped (no valid areas file): {total_skipped_no_valid_areas}")
print(f"Total images checked: {total_with_labels + total_skipped_already_processed + total_skipped_no_geojson + total_skipped_no_valid_areas}")
print(f"\nGeoJSON files used in this run: {len(geojson_cache)}")
if geojson_cache:
    print("Files loaded:")
    for cached_file in geojson_cache.keys():
        cache_path = get_cache_path(cached_file)
        cache_status = "cached on disk" if os.path.exists(cache_path) else "not cached"
        print(f"  - {os.path.basename(cached_file)} ({len(geojson_cache[cached_file])} buildings) - {cache_status}")
print(f"\nPersistent cache directory: {CACHE_DIR}")
print(f"Next run will use cached files (much faster!)")
print(f"\nOutput directory: {base_output_path}")
print("="*70)
    