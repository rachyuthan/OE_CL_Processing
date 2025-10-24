'Compares predictions with baseline. Takes Project ID, containing baseline'
'information, pipeline and image id and outputs three geojson files:'
'1. Baseline_comp_<image_id>.geojson: True Positives'
'2. Baseline_removed_<image_id>.geojson: False Negatives'
'3. Baseline_new_<image_id>.geojson: False Positives'

import geopandas as gpd
import json
from pathlib import Path
import numpy as np
import pyproj
from shapely.geometry import box as shp_box
import pandas as pd
import cv2
import rasterio

# =============================================================================
# GEOJSON-SPECIFIC FUNCTIONS
# The following functions are used ONLY by post_processing_analysis_geojson()
# These handle geographic coordinate operations (EPSG:4326) and GeoJSON I/O
# =============================================================================

def calculate_iou_geometry(geom1, geom2):
    """
    Calculate Intersection over Union (IoU) between two geometries.
    Handles both Shapely geometry objects and bounding box arrays.
    
    [GEOJSON ONLY] Used for comparing actual polygon geometries in geographic coordinates.
    
    Parameters
    ----------
    geom1 : shapely.geometry or array-like
        First geometry (can be Polygon, Point, etc.) or bounding box [x1, y1, x2, y2]
    geom2 : shapely.geometry or array-like
        Second geometry (can be Polygon, Point, etc.) or bounding box [x1, y1, x2, y2]
    
    Returns
    -------
    iou : float
        IoU value between geom1 and geom2
    """
    from shapely.geometry.base import BaseGeometry
    
    # Convert to shapely geometries if they're bounding boxes
    if not isinstance(geom1, BaseGeometry):
        # Assume it's a bounding box [x1, y1, x2, y2]
        geom1 = shp_box(geom1[0], geom1[1], geom1[2], geom1[3])
    
    if not isinstance(geom2, BaseGeometry):
        # Assume it's a bounding box [x1, y1, x2, y2]
        geom2 = shp_box(geom2[0], geom2[1], geom2[2], geom2[3])
    
    # Check if geometries are valid
    if not geom1.is_valid or not geom2.is_valid:
        return 0.0
    
    # Check intersection
    if not geom1.intersects(geom2):
        return 0.0
    
    # Calculate IoU
    try:
        intersection = geom1.intersection(geom2).area
        union = geom1.area + geom2.area - intersection
        if union == 0:
            return 0.0
        return intersection / union
    except:
        # Fallback to 0 if calculation fails
        return 0.0

def save_matching_results_geojson(matched_truths, matched_predictions, removed_indices,
                                   new_indices, truth_gdf, pred_gdf, 
                                   output_dir, image_stem, filtered_distance_indices=None,
                                   filtered_confidence_indices=None):
    """
    [GEOJSON ONLY] Save a combined GeoJSON file with all features color-coded.
    
    All predictions are categorized as follows:
    - Green: Matched (correctly detected buildings)
    - Red: Removed (missed detections)
    - Yellow: New (unfiltered false positives)
    - Orange: New (filtered='Distance') - predictions in buffer zone that didn't match
    - Purple: New (filtered='Confidence') - predictions with low confidence
    
    Note: Filtered predictions maintain type='New' but have a 'filtered' field
    indicating the reason (Distance or Confidence).

    Parameters
    ----------
    matched_truths : set
        Set of matched truth indices
    matched_predictions : set
        Set of matched prediction indices
    removed_indices : list
        List of removed truth indices (false negatives)
    new_indices : list
        List of unmatched prediction indices (false positives)
    truth_gdf : GeoDataFrame
        GeoDataFrame with truth geometries
    pred_gdf : GeoDataFrame
        GeoDataFrame with prediction geometries
    output_dir : str or Path
        Directory to save output files
    image_stem : str
        Image filename stem for naming output files
    filtered_distance_indices : list, optional
        List of prediction indices filtered due to being in buffer zone (not matched)
        These will have type='New' and filtered='Distance'
    filtered_confidence_indices : list, optional
        List of prediction indices filtered due to low confidence
        These will have type='New' and filtered='Confidence'
    
    Returns
    -------
    combined_path : str
        Path to combined GeoJSON file with all features
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_features = []
    tp_count = 0
    fn_count = 0
    fp_count = 0
    filtered_distance_count = 0
    filtered_confidence_count = 0
    
    # Track building IDs to prevent duplicates (when box+point represent same building)
    seen_matched_ids = set()
    seen_removed_ids = set()
    
    # Check if truth data has ID column
    id_column = None
    if truth_gdf is not None and 'id' in truth_gdf.columns:
        id_column = 'id'

    # --- 1. Matched Buildings - GREEN ---
    for truth_idx in matched_truths:
        if truth_idx < len(truth_gdf):
            row = truth_gdf.iloc[truth_idx]
            geom = row.geometry
            
            # Get building ID if available
            building_id = None
            if id_column and id_column in row.index:
                raw_id = row[id_column]
                # Check if ID is valid (not None, NaN, or empty)
                if raw_id is not None and str(raw_id).strip() and str(raw_id).lower() != 'nan':
                    building_id = raw_id
                    
                    # Skip if we've already added this building ID (prevents double-counting)
                    if building_id in seen_matched_ids:
                        continue
                    seen_matched_ids.add(building_id)
            
            feature = {
                'type': 'Feature',
                'properties': {
                    'type': 'Matched',
                    'truth_index': int(truth_idx),
                    'description': 'Correctly detected building'
                },
                'geometry': gpd.GeoSeries([geom]).__geo_interface__['features'][0]['geometry']
            }
            
            # Add building ID to properties if available
            if building_id is not None:
                feature['properties']['building_id'] = str(building_id)
            
            all_features.append(feature)
            tp_count += 1

    # --- 2. Removed Buildings - RED ---
    for truth_idx in removed_indices:
        if truth_idx < len(truth_gdf):
            row = truth_gdf.iloc[truth_idx]
            geom = row.geometry
            
            # Get building ID if available
            building_id = None
            if id_column and id_column in row.index:
                raw_id = row[id_column]
                # Check if ID is valid (not None, NaN, or empty)
                if raw_id is not None and str(raw_id).strip() and str(raw_id).lower() != 'nan':
                    building_id = raw_id
                    
                    # Skip if we've already added this building ID (prevents double-counting)
                    if building_id in seen_removed_ids:
                        continue
                    seen_removed_ids.add(building_id)
            
            feature = {
                'type': 'Feature',
                'properties': {
                    'type': 'Removed',
                    'truth_index': int(truth_idx),
                    'description': 'Removed building (not detected by model)'
                },
                'geometry': gpd.GeoSeries([geom]).__geo_interface__['features'][0]['geometry']
            }
            
            # Add building ID to properties if available
            if building_id is not None:
                feature['properties']['building_id'] = str(building_id)
            
            all_features.append(feature)
            fn_count += 1

    # --- 3. New Buildings - YELLOW ---
    for pred_idx in new_indices:
        if pred_idx < len(pred_gdf):
            geom = pred_gdf.iloc[pred_idx].geometry
            
            # Get confidence score if available
            confidence = None
            if 'confidence' in pred_gdf.columns:
                confidence = float(pred_gdf.iloc[pred_idx]['confidence'])
            
            feature = {
                'type': 'Feature',
                'properties': {
                    'type': 'New',
                    'prediction_index': int(pred_idx),
                    'description': 'New building (no matching ground truth)'
                },
                'geometry': gpd.GeoSeries([geom]).__geo_interface__['features'][0]['geometry']
            }
            
            # Add confidence if available
            if confidence is not None:
                feature['properties']['confidence'] = confidence
            
            all_features.append(feature)
            fp_count += 1
    
    # --- 4. Filtered Distance (predictions in buffer zone that didn't match) - ORANGE ---
    if filtered_distance_indices:
        for pred_idx in filtered_distance_indices:
            if pred_idx < len(pred_gdf):
                geom = pred_gdf.iloc[pred_idx].geometry
                
                # Get confidence score if available
                confidence = None
                if 'confidence' in pred_gdf.columns:
                    confidence = float(pred_gdf.iloc[pred_idx]['confidence'])
                
                feature = {
                    'type': 'Feature',
                    'properties': {
                        'type': 'New',
                        'filtered': 'Distance',
                        'prediction_index': int(pred_idx),
                        'description': 'New building filtered: in buffer zone (outside valid distance, not matched)'
                    },
                    'geometry': gpd.GeoSeries([geom]).__geo_interface__['features'][0]['geometry']
                }
                
                # Add confidence if available
                if confidence is not None:
                    feature['properties']['confidence'] = confidence
                
                all_features.append(feature)
                filtered_distance_count += 1
    
    # --- 5. Filtered Confidence (predictions with low confidence) - PURPLE ---
    if filtered_confidence_indices:
        for pred_idx in filtered_confidence_indices:
            if pred_idx < len(pred_gdf):
                geom = pred_gdf.iloc[pred_idx].geometry
                
                # Get confidence score if available
                confidence = None
                if 'confidence' in pred_gdf.columns:
                    confidence = float(pred_gdf.iloc[pred_idx]['confidence'])
                
                feature = {
                    'type': 'Feature',
                    'properties': {
                        'type': 'New',
                        'filtered': 'Confidence',
                        'prediction_index': int(pred_idx),
                        'description': 'New building filtered: low confidence'
                    },
                    'geometry': gpd.GeoSeries([geom]).__geo_interface__['features'][0]['geometry']
                }
                
                # Add confidence if available
                if confidence is not None:
                    feature['properties']['confidence'] = confidence
                
                all_features.append(feature)
                filtered_confidence_count += 1
    
    # --- 6. CREATE COMBINED GEOJSON ---
    combined_geojson = {
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': 'EPSG:4326'}},
        'properties': {
            'Matched': tp_count,
            'Removed': fn_count,
            'New': fp_count,
            'Filtered_Distance': filtered_distance_count,
            'Filtered_Confidence': filtered_confidence_count,
            'total_features': len(all_features)
        },
        'features': all_features
    }
    
    # --- 7. SAVE COMBINED FILE ---
    combined_path = str(output_dir / f"{image_stem}_baseline_comparison.geojson")
    with open(combined_path, 'w') as f:
        json.dump(combined_geojson, f, indent=2)
    
    print(f"\nSaved combined GeoJSON results:")
    print(f"  File: {combined_path}")
    print(f"  Matched (green): {tp_count} features")
    print(f"  Removed (red): {fn_count} features")
    print(f"  New (yellow): {fp_count} features")
    if filtered_distance_count > 0:
        print(f"  Filtered Distance (orange): {filtered_distance_count} features")
    if filtered_confidence_count > 0:
        print(f"  Filtered Confidence (purple): {filtered_confidence_count} features")
    print(f"  Total: {len(all_features)} features")
    
    return combined_path


# =============================================================================
# END OF GEOJSON-SPECIFIC FUNCTIONS
# =============================================================================

# =============================================================================
# GEOJSON-SPECIFIC HELPER FUNCTIONS (Geographic Coordinate Operations)
# These functions support post_processing_analysis_geojson() 
# All operations are in EPSG:4326 (WGS84) with conversions to EPSG:3857 for metrics
# =============================================================================

def load_geojson_boxes(geojson_input):
    """
    [GEOJSON ONLY] Load boxes from GeoJSON input (file path, dict, or GeoDataFrame) in geographic coordinates.
    
    Parameters
    ----------
    geojson_input : str, Path, dict, list, or GeoDataFrame
        GeoJSON file path, GeoJSON dictionary, list of file paths (for multiple shapefiles), 
        or GeoDataFrame containing building geometries
    
    Returns
    -------
    geo_boxes : numpy array
        Array of bounding boxes in geographic coordinates [minx, miny, maxx, maxy]
        Note: For predictions, these are the actual geometries (boxes).
        For truth data (shapefiles), these are just bounding boxes of potentially complex polygons.
    gdf : GeoDataFrame
        GeoDataFrame with the original geometries for accurate polygon-based matching
    """
    # Load GeoJSON into GeoDataFrame
    if isinstance(geojson_input, list):
        # Handle list of file paths (e.g., multiple shapefiles)
        gdfs = []
        for path in geojson_input:
            gdf_part = gpd.read_file(path)
            gdfs.append(gdf_part)
        # Merge all GeoDataFrames
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        # Set CRS from first file if not set
        if gdf.crs is None and len(gdfs) > 0:
            gdf.crs = gdfs[0].crs
        print(f"Loaded and merged {len(geojson_input)} files: {sum(len(g) for g in gdfs)} total features")
    elif isinstance(geojson_input, (str, Path)):
        gdf = gpd.read_file(geojson_input)
    elif isinstance(geojson_input, dict):
        gdf = gpd.GeoDataFrame.from_features(geojson_input['features'])
        if 'crs' in geojson_input:
            gdf.crs = geojson_input['crs'].get('properties', {}).get('name', 'EPSG:4326')
        else:
            gdf.crs = 'EPSG:4326'
    elif isinstance(geojson_input, gpd.GeoDataFrame):
        gdf = geojson_input.copy()
    else:
        raise ValueError(f"Unsupported geojson_input type: {type(geojson_input)}")
    
    # Ensure CRS is set (default to EPSG:4326 if not specified)
    if gdf.crs is None:
        gdf.crs = 'EPSG:4326'
    
    # Extract bounding boxes from geometries
    # Note: For complex polygons (shapefiles), this simplifies them to rectangular bounds
    # The original geometries are preserved in gdf for accurate IoU calculations
    geo_boxes = []
    for geom in gdf.geometry:
        if geom is not None and not geom.is_empty:
            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            geo_boxes.append(list(bounds))
    
    return np.array(geo_boxes), gdf

def filter_overlapping_boxes(pred_boxes, overlap_threshold=0.7):
    """
    Filter boxes that are mostly contained within larger boxes
    
    Parameters
    -----------
    pred_boxes : list of lists
        List of predicted bounding boxes in the format [x1, y1, x2, y2]
    overlap_threshold : float
        Threshold for overlap ratio (0 to 1)
    
    Returns
    --------
    filtered_indices : set
        Set of indices of boxes that are filtered out
    """
    n = len(pred_boxes)
    filtered_indices = set()
    
    # Sort boxes by area (largest first)
    areas = [(i, (box[2]-box[0])*(box[3]-box[1])) for i, box in enumerate(pred_boxes)]
    areas.sort(key=lambda x: x[1], reverse=True)
    
    for i, (idx1, area1) in enumerate(areas):
        if idx1 in filtered_indices:
            continue
        box1 = pred_boxes[idx1]
        
        for idx2, area2 in areas[i+1:]:
            if idx2 in filtered_indices:
                continue
            box2 = pred_boxes[idx2]
            
            # Calculate intersection
            intersection = calculate_iou_geometry(box1, box2) 
            smaller_area = min(area1, area2)
            overlap_ratio = intersection * (area1 + area2) / smaller_area
            
            if overlap_ratio >= overlap_threshold:
                if idx1 not in filtered_indices:
                    filtered_indices.add(idx1) 
    
    return filtered_indices

def filter_geojson_boxes_by_overlap(geo_boxes, gdf, overlap_threshold=0.7):
    """
    [GEOJSON ONLY] Filter overlapping boxes in geographic coordinates using NMS-style approach.
    
    Parameters
    ----------
    geo_boxes : numpy array
        Array of bounding boxes in geographic coordinates [minx, miny, maxx, maxy]
    gdf : GeoDataFrame
        GeoDataFrame with the geometries
    overlap_threshold : float
        IoU threshold for filtering overlapping boxes (0 to 1)
    
    Returns
    -------
    filtered_geo_boxes : numpy array
        Filtered array of bounding boxes in geographic coordinates
    filtered_gdf : GeoDataFrame
        Filtered GeoDataFrame
    """
    if len(geo_boxes) == 0:
        return geo_boxes, gdf
    
    # Use existing filter_overlapping_boxes function
    filtered_indices = filter_overlapping_boxes(geo_boxes, overlap_threshold=overlap_threshold)
    valid_indices = [i for i in range(len(geo_boxes)) if i not in filtered_indices]
    
    filtered_geo_boxes = geo_boxes[valid_indices]
    filtered_gdf = gdf.iloc[valid_indices].reset_index(drop=True)
    
    return filtered_geo_boxes, filtered_gdf

def classify_geo_boxes(geo_boxes, gdf=None, area_threshold=200, crs='EPSG:4326'):
    """
    [GEOJSON ONLY] Separates geographic boxes into points and regular boxes based on geometry type.
    
    Parameters
    ----------
    geo_boxes : numpy array
        Array of bounding boxes in geographic coordinates [minx, miny, maxx, maxy]
    gdf : GeoDataFrame, optional
        GeoDataFrame with the geometries. If provided, uses actual geometry type
        (Point vs Polygon/MultiPolygon) for classification. This is more accurate
        than area-based classification.
    area_threshold : float
        Area threshold in square meters for classifying as points (fallback if gdf not provided)
    crs : str or pyproj.CRS
        Coordinate reference system of the boxes
    
    Returns
    -------
    point_indices : list of int
        List of indices of points
    box_indices : list of int
        List of indices of regular boxes
    """
    point_indices = []
    box_indices = []
    
    # If GeoDataFrame provided, use actual geometry type (most accurate)
    if gdf is not None:
        from shapely.geometry import Point, MultiPoint
        
        for i, geom in enumerate(gdf.geometry):
            if isinstance(geom, (Point, MultiPoint)):
                point_indices.append(i)
            else:
                # Polygon, MultiPolygon, LineString, etc. are treated as boxes
                box_indices.append(i)
        
        print(f"Classified by geometry type: {len(point_indices)} points, {len(box_indices)} polygons")
        return point_indices, box_indices
    
    # Fallback: Area-based classification (less accurate, kept for backward compatibility)
    print("Warning: GeoDataFrame not provided, using area-based classification (less accurate)")
    
    # Create transformer to a metric CRS for area calculation if needed
    source_crs = pyproj.CRS(crs)
    if source_crs.is_geographic:
        # Use UTM or a suitable metric projection
        # For simplicity, use Web Mercator (EPSG:3857)
        target_crs = pyproj.CRS.from_epsg(3857)
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    else:
        transformer = None
    
    for i, box in enumerate(geo_boxes):
        minx, miny, maxx, maxy = box
        
        if transformer:
            # Transform to metric CRS for accurate area calculation
            minx_m, miny_m = transformer.transform(minx, miny)
            maxx_m, maxy_m = transformer.transform(maxx, maxy)
            width = maxx_m - minx_m
            height = maxy_m - miny_m
        else:
            # Already in metric CRS
            width = maxx - minx
            height = maxy - miny
        
        area = width * height
        
        if area < area_threshold:
            point_indices.append(i)
        else:
            box_indices.append(i)
    
    return point_indices, box_indices

def calculate_geo_distance(point_geo, box_geo):
    """
    [GEOJSON ONLY] Calculate distance between a point and a box in geographic coordinates (meters).
    
    Parameters
    ----------
    point_geo : array-like
        Point coordinates [minx, miny, maxx, maxy] (for small boxes representing points)
    box_geo : array-like
        Box coordinates [minx, miny, maxx, maxy]
    
    Returns
    -------
    distance : float
        Distance in meters
    """
    # Calculate center of point
    point_center_x = (point_geo[0] + point_geo[2]) / 2
    point_center_y = (point_geo[1] + point_geo[3]) / 2
    
    # Calculate center of box
    box_center_x = (box_geo[0] + box_geo[2]) / 2
    box_center_y = (box_geo[1] + box_geo[3]) / 2
    
    # Use shapely to calculate geodesic distance
    from shapely.geometry import Point
    
    point1 = Point(point_center_x, point_center_y)
    point2 = Point(box_center_x, box_center_y)
    
    # For more accurate distance, use geodesic calculation
    # Simplified approach using pyproj for metric projection
    try:
        # Transform to Web Mercator for approximate metric distance
        wgs84 = pyproj.CRS('EPSG:4326')
        web_merc = pyproj.CRS('EPSG:3857')
        transformer = pyproj.Transformer.from_crs(wgs84, web_merc, always_xy=True)
        
        x1, y1 = transformer.transform(point_center_x, point_center_y)
        x2, y2 = transformer.transform(box_center_x, box_center_y)
        
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    except:
        # Fallback to simple Euclidean in degrees (very approximate)
        distance = np.sqrt((box_center_x - point_center_x)**2 + (box_center_y - point_center_y)**2) * 111320
    
    return distance

def find_geo_box_point_pairs(geo_boxes, point_indices, box_indices, distance_threshold=10, gdf=None):
    """
    [GEOJSON ONLY] Identifies points that are close to boxes in geographic coordinates.
    If GeoDataFrame has an 'id' attribute, uses ID matching for more accuracy.
    
    Parameters
    ----------
    geo_boxes : numpy array
        Array of bounding boxes in geographic coordinates [minx, miny, maxx, maxy]
    point_indices : list of int
        List of indices of points
    box_indices : list of int
        List of indices of regular boxes
    distance_threshold : float
        Distance threshold in meters for pairing points with boxes
    gdf : GeoDataFrame, optional
        GeoDataFrame with the geometries. If provided and contains 'id' field,
        uses ID-based matching (more accurate for shapefiles)
    
    Returns
    -------
    box_point_pairs : dict
        Dictionary mapping box indices to lists of point indices
    remaining_points : list of int
        List of point indices that are not associated with any box
    standalone_boxes : list of int
        List of box indices that don't have any associated points
    """
    box_point_pairs = {}
    points_to_remove = set()
    
    # Check if we can use ID-based matching (more accurate for shapefiles)
    use_id_matching = False
    id_column = None
    if gdf is not None:
        # Look for ID name (case-insensitive)
        potential_id_cols = ['id'] # Add more potential ID column names if needed
        for col in potential_id_cols:
            if col in gdf.columns:
                id_column = col
                use_id_matching = True
                print(f"Using ID-based matching with column: '{id_column}'")
                break
    
    if use_id_matching and id_column:
        # ID-BASED MATCHING (more accurate for shapefiles)
        # Build ID lookup for points and boxes
        point_ids = {p_idx: gdf.iloc[p_idx][id_column] for p_idx in point_indices}
        box_ids = {b_idx: gdf.iloc[b_idx][id_column] for b_idx in box_indices}
        
        # Match points to boxes by ID
        for p_idx, p_id in point_ids.items():
            for b_idx, b_id in box_ids.items():
                if p_id == b_id:  # Same ID means they represent the same feature
                    # Associate this point with this box
                    if b_idx not in box_point_pairs:
                        box_point_pairs[b_idx] = []
                    box_point_pairs[b_idx].append(p_idx)
                    points_to_remove.add(p_idx)
                    break
    else:
        # DISTANCE-BASED MATCHING (fallback for data without IDs)
        for p_idx in point_indices:
            point = geo_boxes[p_idx]
            
            for b_idx in box_indices:
                box = geo_boxes[b_idx]
                
                # Calculate distance between point and box
                distance = calculate_geo_distance(point, box)
                
                if distance <= distance_threshold:
                    # Associate this point with this box
                    if b_idx not in box_point_pairs:
                        box_point_pairs[b_idx] = []
                    box_point_pairs[b_idx].append(p_idx)
                    points_to_remove.add(p_idx)
                    break
    
    # Remove paired points from point_indices
    remaining_points = [p for p in point_indices if p not in points_to_remove]
    
    # Identify standalone boxes (boxes without associated points)
    standalone_boxes = [b_idx for b_idx in box_indices if b_idx not in box_point_pairs]
    
    return box_point_pairs, remaining_points, standalone_boxes

def create_pipeline_buffer_gdf(pipeline_shp_path, max_distance_meters, target_crs='EPSG:4326'):
    """
    [GEOJSON ONLY] Create a GeoDataFrame with pipeline buffer for filtering/intersection checks.
    Uses UTM projection for accurate distance-based buffering.
    
    Parameters
    ----------
    pipeline_shp_path : str or Path
        Path to pipeline shapefile
    max_distance_meters : float
        Maximum distance in meters for buffer zone
    target_crs : str
        Target CRS for output (default: EPSG:4326)
    
    Returns
    -------
    pipeline_buffer_gdf : GeoDataFrame
        GeoDataFrame with pipeline buffer geometry in target CRS
    metric_crs : str
        The metric CRS used for buffer calculation (for reference)
    """
    # Load pipeline shapefile
    pipeline_gdf = gpd.read_file(pipeline_shp_path)
    
    # Ensure in target CRS
    if pipeline_gdf.crs != target_crs:
        pipeline_gdf = pipeline_gdf.to_crs(target_crs)
    
    # Get the centroid of the pipeline to determine best UTM zone
    # This gives more accurate distance measurements than Web Mercator
    pipeline_centroid = pipeline_gdf.union_all().centroid
    lon, lat = pipeline_centroid.x, pipeline_centroid.y
    
    # Calculate appropriate UTM zone
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'
    
    # Try to use UTM projection for better accuracy
    try:
        if hemisphere == 'north':
            metric_crs = f'EPSG:326{utm_zone:02d}'  # UTM North
        else:
            metric_crs = f'EPSG:327{utm_zone:02d}'  # UTM South
        
        # Test if this CRS is valid
        pipeline_gdf_metric = pipeline_gdf.to_crs(metric_crs)
        print(f"Pipeline buffer using {metric_crs} (UTM Zone {utm_zone}{hemisphere[0].upper()}) for accurate {max_distance_meters}m buffer")
    except:
        # Fallback to Web Mercator if UTM fails
        metric_crs = 'EPSG:3857'
        pipeline_gdf_metric = pipeline_gdf.to_crs(metric_crs)
        print(f"Pipeline buffer using EPSG:3857 (Web Mercator) for {max_distance_meters}m buffer")
    
    # Create buffer around pipeline in meters
    pipeline_buffer_metric = pipeline_gdf_metric.buffer(max_distance_meters).union_all()
    
    # Transform buffer back to target CRS
    pipeline_buffer_gdf = gpd.GeoDataFrame(
        {'geometry': [pipeline_buffer_metric]},
        crs=metric_crs
    ).to_crs(target_crs)
    
    return pipeline_buffer_gdf, metric_crs

def filter_predictions_by_pipeline(pred_gdf, pipeline_shp_path, max_distance_meters, buffer, target_crs='EPSG:4326'):
    """
    [GEOJSON ONLY] Filter predictions based on distance to pipeline using geographic operations.
    Uses a two-tier system:
    - Within max_distance: Keep for matching (valid predictions)
    - Within max_distance + buffer: Keep for matching, but mark as "in buffer zone"
    - Beyond max_distance + buffer: Reject immediately
    
    Predictions in the buffer zone will be re-evaluated later - if they end up as "New" (unmatched),
    they'll be moved to a "Filtered: Distance" category instead.
    
    Parameters
    ----------
    pred_gdf : GeoDataFrame
        GeoDataFrame with prediction geometries
    pipeline_shp_path : str or Path
        Path to pipeline shapefile
    max_distance_meters : float
        Maximum distance in meters for valid predictions
    buffer : float
        Additional buffer distance in meters (grace period)
    target_crs : str
        Target CRS (default: EPSG:4326)
    
    Returns
    -------
    filtered_gdf : GeoDataFrame
        Filtered GeoDataFrame with predictions within max_distance + buffer
        Includes 'in_buffer_zone' column marking predictions in the buffer zone
    rejected_count : int
        Number of predictions rejected (beyond max_distance + buffer)
    """
    # Create two buffers: one for valid zone, one for grace zone
    valid_buffer_gdf, metric_crs = create_pipeline_buffer_gdf(
        pipeline_shp_path, max_distance_meters, target_crs
    )
    grace_buffer_gdf, _ = create_pipeline_buffer_gdf(
        pipeline_shp_path, max_distance_meters + buffer, target_crs
    )
    
    valid_buffer_geom = valid_buffer_gdf.geometry.iloc[0]
    grace_buffer_geom = grace_buffer_gdf.geometry.iloc[0]
    
    # Check which predictions are within each buffer
    within_valid_zone = pred_gdf.geometry.intersects(valid_buffer_geom)
    within_grace_zone = pred_gdf.geometry.intersects(grace_buffer_geom)
    
    # Keep predictions within grace zone (includes both valid + buffer zones)
    filtered_gdf = pred_gdf[within_grace_zone].copy()
    
    # Mark predictions that are ONLY in buffer zone (not in valid zone)
    # These will be re-evaluated later
    in_buffer_only = within_grace_zone & ~within_valid_zone
    filtered_gdf['in_buffer_zone'] = in_buffer_only[within_grace_zone].values
    
    rejected_count = len(pred_gdf) - len(filtered_gdf)
    buffer_count = filtered_gdf['in_buffer_zone'].sum()
    
    print(f"Pipeline filtering:")
    print(f"  Valid zone (≤{max_distance_meters}m): {(~filtered_gdf['in_buffer_zone']).sum()} predictions")
    print(f"  Buffer zone ({max_distance_meters}m-{max_distance_meters + buffer}m): {buffer_count} predictions")
    print(f"  Rejected (>{max_distance_meters + buffer}m): {rejected_count} predictions")
    print(f"  Total kept: {len(filtered_gdf)} predictions")
    
    return filtered_gdf, rejected_count

# =============================================================================
# END OF GEOJSON-SPECIFIC HELPER FUNCTIONS
# =============================================================================



# =============================================================================
# GEOJSON-SPECIFIC MATCHING FUNCTIONS (Geographic Coordinate Matching)
# These functions perform the actual matching logic in geographic coordinates
# =============================================================================

def match_geo_box_point_pairs(box_point_pairs, geo_truth_boxes, geo_pred_boxes, 
                                matched_truths, matched_predictions,
                                truth_gdf=None, pred_gdf=None):
    """
    [GEOJSON ONLY] Process and match box-point pairs to predictions using geographic coordinates.
    Uses actual polygon geometries for IoU calculation when available.
    
    Parameters
    ----------
    box_point_pairs : dict
        Dictionary mapping box indices to lists of point indices
    geo_truth_boxes : numpy array
        Array of truth bounding boxes in geographic coordinates [minx, miny, maxx, maxy]
    geo_pred_boxes : numpy array
        Array of predicted bounding boxes in geographic coordinates [minx, miny, maxx, maxy]
    matched_truths : set
        Set of matched truth indices
    matched_predictions : set
        Set of matched prediction indices
    truth_gdf : GeoDataFrame, optional
        GeoDataFrame with original truth geometries (for accurate polygon-based IoU)
    pred_gdf : GeoDataFrame, optional
        GeoDataFrame with prediction geometries
    
    Returns
    -------
    processed_pairs : set
        Set of processed point indices
    """
    processed_pairs = set()
    
    for b_idx, p_indices in box_point_pairs.items():
        box = geo_truth_boxes[b_idx]
        max_iou = 0
        best_match = -1
        
        # Get the actual truth geometry if available (more accurate for polygons)
        truth_geom = truth_gdf.iloc[b_idx].geometry if truth_gdf is not None else box
        
        # Find best prediction match for this box
        for j, pred_box in enumerate(geo_pred_boxes):
            if j not in matched_predictions:
                # Get prediction geometry (usually a box, but could be from GeoDataFrame)
                pred_geom = pred_gdf.iloc[j].geometry if pred_gdf is not None else pred_box
                
                # Use geometry-aware IoU calculation
                iou = calculate_iou_geometry(truth_geom, pred_geom)
                if iou > max_iou:
                    max_iou = iou
                    best_match = j
        
        # If box is matched (using relaxed threshold), consider it a full match
        if max_iou >= 0.1:  # Lower threshold but counted as full match
            # ONLY add the box index (not paired points) to prevent double-counting
            matched_truths.add(b_idx)
            matched_predictions.add(best_match)
            
            # Track processed pairs but DON'T add them to matched_truths
            # (they represent the same building, so shouldn't be counted separately)
            for p_idx in p_indices:
                processed_pairs.add(p_idx)
    
    return processed_pairs

def match_geo_standalone_boxes(box_indices, geo_truth_boxes, geo_pred_boxes, 
                                 matched_truths, matched_predictions,
                                 truth_gdf=None, pred_gdf=None):
    """
    [GEOJSON ONLY] Match standalone boxes (no associated points) to predictions using geographic coordinates.
    Uses actual polygon geometries for IoU calculation when available.
    
    Parameters
    ----------
    box_indices : list of int
        List of indices of box indices
    geo_truth_boxes : numpy array
        Array of truth bounding boxes in geographic coordinates [minx, miny, maxx, maxy]
    geo_pred_boxes : numpy array
        Array of predicted bounding boxes in geographic coordinates [minx, miny, maxx, maxy]
    matched_truths : set
        Set of matched truth indices
    matched_predictions : set
        Set of matched prediction indices
    truth_gdf : GeoDataFrame, optional
        GeoDataFrame with original truth geometries (for accurate polygon-based IoU)
    pred_gdf : GeoDataFrame, optional
        GeoDataFrame with prediction geometries
    """
    for i in box_indices:
        if i in matched_truths:
            continue  # Skip already matched boxes
            
        box = geo_truth_boxes[i]
        max_iou = 0
        best_match = -1
        
        # Get the actual truth geometry if available
        truth_geom = truth_gdf.iloc[i].geometry if truth_gdf is not None else box
        
        # Find best prediction match
        for j, pred_box in enumerate(geo_pred_boxes):
            if j not in matched_predictions:
                # Get prediction geometry
                pred_geom = pred_gdf.iloc[j].geometry if pred_gdf is not None else pred_box
                
                # Use geometry-aware IoU calculation
                iou = calculate_iou_geometry(truth_geom, pred_geom)
                if iou > max_iou:
                    max_iou = iou
                    best_match = j
        
        # If good match found (using relaxed threshold)
        if max_iou >= 0.3:
            matched_truths.add(i)
            matched_predictions.add(best_match)

def find_potential_geo_point_matches(point_indices, geo_truth_boxes, geo_pred_boxes, 
                                      processed_pairs, matched_truths, matched_predictions, 
                                      point_distance_tolerance, pred_confidences=None):
    """
    [GEOJSON ONLY] Find all potential matches between points and predictions using geographic coordinates.
    
    Parameters
    ----------
    point_indices : list of int
        List of indices of points
    geo_truth_boxes : numpy array
        Array of truth bounding boxes in geographic coordinates [minx, miny, maxx, maxy]
    geo_pred_boxes : numpy array
        Array of predicted bounding boxes in geographic coordinates [minx, miny, maxx, maxy]
    processed_pairs : set
        Set of processed point indices
    matched_truths : set
        Set of matched truth indices
    matched_predictions : set
        Set of matched prediction indices
    point_distance_tolerance : float
        Distance tolerance in meters for matching points to predictions
    pred_confidences : list, optional
        List of confidence scores for predictions
    
    Returns
    -------
    point_to_pred_matches : dict
        Dictionary mapping point indices to lists of potential prediction matches
    """
    point_to_pred_matches = {}  # Maps point index to [(pred_idx, distance, confidence), ...]

    for i in point_indices:
        if i in processed_pairs or i in matched_truths:
            continue  # Skip points already processed
            
        truth_point = geo_truth_boxes[i]
        point_center_x = (truth_point[0] + truth_point[2]) / 2
        point_center_y = (truth_point[1] + truth_point[3]) / 2
        
        # Track all potential matches for this point
        potential_matches = []
        
        # Check all prediction boxes for potential matches
        for j, pred_box in enumerate(geo_pred_boxes):
            if j in matched_predictions:
                continue
            
            current_confidence = pred_confidences[j] if pred_confidences is not None and j < len(pred_confidences) else 0.0
                
            pred_center_x = (pred_box[0] + pred_box[2]) / 2
            pred_center_y = (pred_box[1] + pred_box[3]) / 2
            
            # Check if point is inside box
            if (pred_box[0] <= point_center_x <= pred_box[2] and 
                pred_box[1] <= point_center_y <= pred_box[3]):
                # Calculate distance to center in meters
                distance = calculate_geo_distance(truth_point, pred_box)
                potential_matches.append((j, distance, current_confidence))
                
            # Or check if point is NEAR box (using distance tolerance)
            else:
                # Calculate closest distance to box
                distance = calculate_geo_distance(truth_point, pred_box)
                
                if distance <= point_distance_tolerance:
                    potential_matches.append((j, distance, current_confidence))
        
        if potential_matches:
            point_to_pred_matches[i] = potential_matches

    return point_to_pred_matches

def assign_geo_point_matches(point_matches, matched_truths, matched_predictions):
    """
    [GEOJSON ONLY] Assign optimal matches between points and predictions using geographic coordinates.
    
    Parameters
    ----------
    point_matches : dict
        Dictionary mapping point indices to lists of potential prediction matches
    matched_truths : set
        Set of matched truth indices
    matched_predictions : set
        Set of matched prediction indices
    """
    # Sort all point indices by number of potential matches (ascending)
    sorted_points = sorted(point_matches.keys(), 
                        key=lambda x: len(point_matches[x]))

    for point_idx in sorted_points:
        if point_idx in matched_truths:
            continue  # Skip if already matched in a previous iteration
            
        # Sort potential matches by confidence (descending), then by distance (ascending) as a tie-breaker
        matches = sorted(point_matches[point_idx], key=lambda x: (x[2], -x[1]), reverse=True)
        
        for pred_idx, distance, confidence in matches:
            if pred_idx not in matched_predictions:
                # This is the best available match
                matched_truths.add(point_idx)
                matched_predictions.add(pred_idx)
                break  # Stop after finding the best match

# =============================================================================
# END OF GEOJSON-SPECIFIC MATCHING FUNCTIONS
# =============================================================================

# ============================================================================
# IMAGE PREPERATION FUNCTIONS
# ============================================================================

def load_and_prepare_image(image_path):
    """
    Loads image and extracts relevant metadata. Assumes input is .tif or .tiff for GeoTIFF.
    
    Parameters
    ----------
    image_path : str
        Path to the image file
    
    Returns
    -------
    image : numpy array
        Loaded image
    transform : pyproj.Transformer
        Transformer for converting pixel coordinates to geographic coordinates
    is_geotiff : bool
        Whether the image is a GeoTIFF
    bounds_geo : tuple
        Geographic bounds of the image (left, bottom, right, top)
    """
    image_path_str = str(image_path)
    is_geotiff = image_path_str.lower().endswith('.tif') or image_path_str.lower().endswith('.tiff')
    
    if is_geotiff:
        with rasterio.open(image_path) as src:
            image = src.read().transpose(1, 2, 0)
            if image.shape[2] >= 3:
                image = image[:, :, [2, 1, 0]]
            
            transform = src.transform
            bounds_geo = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        transform, bounds_geo = None, None

    return image, transform, is_geotiff, bounds_geo



# =============================================================================
# GEOJSON-SPECIFIC POST-PROCESSING FUNCTION
# This is the main GeoJSON-based analysis function (alternative to post_processing_analysis)
# =============================================================================

def baseline_comparison_geo(pred_geojson, truth_geojson, image_path, output_dir, 
                                      pipeline_shp_path=None,
                                      max_distance=100,
                                      buffer=50,
                                      point_distance_tolerance=10,
                                      pred_confidences=None, 
                                      save_images=False):
    """
    [GEOJSON ONLY] Analyzes building detections using GeoJSON inputs in geographic coordinates.
    Compares predictions against ground truth with special handling for points and boxes.
    
    Parameters
    ----------
    pred_geojson : str, Path, dict, or GeoDataFrame
        Predictions as GeoJSON file path, dictionary, or GeoDataFrame
    truth_geojson : str, Path, dict, list, or GeoDataFrame
        Ground truth (baseline) as GeoJSON file path, dictionary, list of file paths 
        (for multiple shapefiles - e.g., [points.shp, polygons.shp]), or GeoDataFrame
    image_path : str or Path
        Path to image file (for visualization and coordinate transforms)
    output_dir : str
        Directory to save output
    pipeline_shp_path : str, optional
        Path to pipeline shapefile for filtering predictions
    max_distance : float
        Maximum distance in meters for valid predictions from pipeline (default: 100)
    buffer : float
        Additional buffer distance in meters for grace zone (default: 50)
        Predictions in buffer zone (max_distance to max_distance+buffer) are kept for matching,
        but if they remain unmatched, they're marked as "Filtered: Distance" instead of "New"
    point_distance_tolerance : float
        Distance tolerance for point matching in meters
    pred_confidences : list, optional
        List of confidence scores for predictions (used for optimal point matching)
    save_images : bool
        Whether to save annotated images (default: False)
    
    Returns
    -------
    combined_geojson_path : str
        Path to combined GeoJSON file containing all features with properties:
        - type='Matched', color=green: Correctly detected buildings
        - type='Removed', color=red: Missed buildings
        - type='New', color=yellow: Unfiltered false detections (high confidence, valid zone)
        - type='New', filtered='Distance', color=orange: Unmatched predictions in buffer zone
        
        Note: Confidence filtering is handled separately by post_processing.py
        which adds filtered='Confidence' field to low-confidence predictions (purple color)
    """
    # --- 1. INITIALIZATION AND IMAGE LOADING ---
    image_path = Path(image_path)
    image, transform, is_geotiff, bounds_geo = load_and_prepare_image(image_path)
    annotated_image = image.copy() if save_images else None
    
    # --- 2. LOAD GEOJSON BOXES IN GEOGRAPHIC COORDINATES ---
    geo_pred_boxes, pred_gdf = load_geojson_boxes(pred_geojson)
    
    # Load truth GeoDataFrame (don't extract boxes yet - will do after filtering)
    if isinstance(truth_geojson, list):
        # Handle list of file paths (e.g., multiple shapefiles)
        gdfs = []
        for path in truth_geojson:
            gdf_part = gpd.read_file(path)
            gdfs.append(gdf_part)
        # Merge all GeoDataFrames
        truth_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        # Set CRS from first file if not set
        if truth_gdf.crs is None and len(gdfs) > 0:
            truth_gdf.crs = gdfs[0].crs
        print(f"Loaded and merged {len(truth_geojson)} files: {sum(len(g) for g in gdfs)} total features")
    elif isinstance(truth_geojson, (str, Path)):
        truth_gdf = gpd.read_file(truth_geojson)
    elif isinstance(truth_geojson, dict):
        truth_gdf = gpd.GeoDataFrame.from_features(truth_geojson['features'])
        if 'crs' in truth_geojson:
            truth_gdf.crs = truth_geojson['crs'].get('properties', {}).get('name', 'EPSG:4326')
        else:
            truth_gdf.crs = 'EPSG:4326'
    elif isinstance(truth_geojson, gpd.GeoDataFrame):
        truth_gdf = truth_geojson.copy()
    else:
        raise ValueError(f"Unsupported truth_geojson type: {type(truth_geojson)}")
    
    # Ensure CRS is set (default to EPSG:4326 if not specified)
    if truth_gdf.crs is None:
        truth_gdf.crs = 'EPSG:4326'
    
    # IMPORTANT: All data is standardized to EPSG:4326 (WGS84 lat/lon)
    # This is the standard geographic coordinate system for GeoJSON
    # All matching, distance calculations, and IoU are done in this CRS
    # (Functions will transform to Web Mercator EPSG:3857 internally for metric calculations)
    target_crs = 'EPSG:4326'
    
    # Convert prediction GeoDataFrame to EPSG:4326 if needed
    if pred_gdf.crs != target_crs:
        print(f"Converting predictions from {pred_gdf.crs} to {target_crs}")
        pred_gdf = pred_gdf.to_crs(target_crs)
        # Re-extract bounding boxes after CRS transformation
        geo_pred_boxes = np.array([list(geom.bounds) for geom in pred_gdf.geometry if geom is not None and not geom.is_empty])
    
    # Filter overlapping predictions
    geo_pred_boxes, pred_gdf = filter_geojson_boxes_by_overlap(geo_pred_boxes, pred_gdf, overlap_threshold=0.7)
    
    # Convert truth GeoDataFrame to EPSG:4326 if needed
    if truth_gdf.crs != target_crs:
        print(f"Converting truth from {truth_gdf.crs} to {target_crs}")
        truth_gdf = truth_gdf.to_crs(target_crs)
    
    # --- 2a. FILTER TRUTH GEOJSON TO IMAGE BOUNDS (BEFORE extracting boxes) ---
    # Only consider truth geometries that intersect with the image bounds
    if is_geotiff and bounds_geo is not None:
        image_bounds_polygon = shp_box(bounds_geo[0], bounds_geo[1], bounds_geo[2], bounds_geo[3])
        
        # Filter truth GeoDataFrame to only include geometries within/intersecting image bounds
        # This is done BEFORE extracting bounding boxes for efficiency
        truth_within_bounds = truth_gdf.geometry.intersects(image_bounds_polygon)
        truth_gdf = truth_gdf[truth_within_bounds].reset_index(drop=True)
        
        print(f"Filtered truth data: {len(truth_gdf)} geometries within image bounds")
    
    # Extract bounding boxes from the filtered truth GeoDataFrame
    
    geo_truth_boxes = np.array([list(geom.bounds) for geom in truth_gdf.geometry if geom is not None and not geom.is_empty])
    
    # --- 3. PIPELINE FILTERING (if applicable) ---
    if pipeline_shp_path:
        # Filter predictions using accurate UTM-based distance filtering with buffer zone
        original_count = len(pred_gdf)
        pred_gdf, rejected_count = filter_predictions_by_pipeline(
            pred_gdf, pipeline_shp_path, max_distance, buffer, target_crs=target_crs
        )
        
        # Re-extract bounding boxes from filtered predictions
        geo_pred_boxes = np.array([list(geom.bounds) for geom in pred_gdf.geometry if geom is not None and not geom.is_empty])
        
        # Update confidences if provided
        if pred_confidences is not None and rejected_count > 0:
            # Keep only confidences for remaining predictions
            # Since we don't know which specific indices were removed, we need to be careful
            # The filtered pred_gdf has been reset, so we'll need to truncate or rebuild
            if len(pred_confidences) == original_count:
                # This is tricky - we'd need to track indices. For now, just warn.
                print(f"Warning: Confidences list may not align after pipeline filtering")
                # Safest is to keep the first N confidences where N = len(pred_gdf)
                pred_confidences = pred_confidences[:len(pred_gdf)] if len(pred_confidences) >= len(pred_gdf) else pred_confidences
    
    # --- 4. CLASSIFICATION OF TRUTH BOXES ---
    # Classify using actual geometry type from GeoDataFrame (Point vs Polygon)
    # This is more accurate than area-based classification
    point_indices, box_indices = classify_geo_boxes(geo_truth_boxes, gdf=truth_gdf, crs=target_crs)

    # --- 5. MATCHING PROCESS ---
    matched_predictions, matched_truths = set(), set()
    
    # Phase 1: Find box-point pairs
    box_point_pairs, point_indices, standalone_box_indices = find_geo_box_point_pairs(
        geo_truth_boxes, point_indices, box_indices, distance_threshold=point_distance_tolerance, gdf=truth_gdf
    )
    
    # Phase 2: Process box-point pairs
    processed_pairs = match_geo_box_point_pairs(
        box_point_pairs, geo_truth_boxes, geo_pred_boxes, 
        matched_truths, matched_predictions,
        truth_gdf=truth_gdf, pred_gdf=pred_gdf
    )
    
    # Phase 3: Match standalone boxes (boxes without associated points)
    match_geo_standalone_boxes(
        standalone_box_indices, geo_truth_boxes, geo_pred_boxes,
        matched_truths, matched_predictions,
        truth_gdf=truth_gdf, pred_gdf=pred_gdf
    )
    
    # Phase 4: Match standalone points
    point_matches = find_potential_geo_point_matches(
        point_indices, geo_truth_boxes, geo_pred_boxes,
        processed_pairs, matched_truths, matched_predictions, 
        point_distance_tolerance, pred_confidences=pred_confidences
    )
    
    # Assign optimal point matches
    assign_geo_point_matches(
        point_matches, matched_truths, matched_predictions
    )
    
    # --- 6. IDENTIFY MISSED DETECTIONS ---
    missed_points = [i for i in point_indices if i not in matched_truths]
    missed_boxes = [i for i in box_indices if i not in matched_truths]
    
    # --- 7. COLLECT FALSE POSITIVES AND FILTER BY BUFFER ZONE ---
    # Separate unmatched predictions into categories:
    # - "New" (false positives in valid zone)
    # - "Filtered: Distance" (unmatched predictions in buffer zone)
    
    all_unmatched = [i for i in range(len(geo_pred_boxes)) if i not in matched_predictions]
    
    false_positive_indices = []
    filtered_distance_indices = []
    
    # Filter by distance (buffer zone)
    if pipeline_shp_path and 'in_buffer_zone' in pred_gdf.columns:
        for idx in all_unmatched:
            if pred_gdf.iloc[idx]['in_buffer_zone']:
                # Prediction is in buffer zone and didn't match -> filter by distance
                filtered_distance_indices.append(idx)
            else:
                # Prediction is in valid zone -> mark as "New"
                false_positive_indices.append(idx)
    else:
        # No buffer zone filtering - all unmatched are "New"
        false_positive_indices = all_unmatched
    
    # Collect missed indices (false negatives)
    missed_indices = missed_points + missed_boxes
    
    # --- 8. SAVE COMBINED GEOJSON RESULTS ---
    # Save single combined GeoJSON file with all features color-coded
    # Note: Confidence filtering is done separately in post_processing.py
    combined_geojson_path = save_matching_results_geojson(
        matched_truths=matched_truths,
        matched_predictions=matched_predictions,
        removed_indices=missed_indices,
        new_indices=false_positive_indices,
        truth_gdf=truth_gdf,
        pred_gdf=pred_gdf,
        output_dir=output_dir,
        image_stem=image_path.stem,
        filtered_distance_indices=filtered_distance_indices,
        filtered_confidence_indices=[]  # Empty - confidence filtering done in post_processing.py
    )

    # --- 9. SAVE IMAGE VISUALIZATION (OPTIONAL) ---
    if save_images and annotated_image is not None:
        output_path = str(Path(output_dir) / f"{image_path.stem}_analysis.jpg")
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved annotated image: {output_path}")
    
    return combined_geojson_path