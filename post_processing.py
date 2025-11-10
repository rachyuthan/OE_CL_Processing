"""
Post-processing functions for GeoJSON-based building detection analysis.

This module provides functions for filtering and processing GeoJSON results
from baseline comparison, including confidence-based false positive filtering
and building type filtering.
"""

import geopandas as gpd
import json
from pathlib import Path
from shapely.geometry import shape, mapping, box
import pandas as pd
import rasterio


def get_image_bounds(image_path=None, geojson_path=None):
    """
    Extract geographic bounds from either a GeoTIFF image or GeoJSON file.
    
    This is a helper function for spatial filtering optimization. Returns bounds
    in EPSG:4326 (WGS84) coordinate system for consistency with GeoJSON.
    
    Parameters
    ----------
    image_path : str or Path, optional
        Path to GeoTIFF image file
    geojson_path : str or Path, optional
        Path to GeoJSON file (alternative if no image)
    
    Returns
    -------
    bounds : tuple
        Geographic bounds (minx, miny, maxx, maxy) in EPSG:4326
        Returns None if bounds cannot be determined
    
    Notes
    -----
    Priority: image_path is used if provided, otherwise geojson_path.
    For GeoTIFF: Extracts bounds from raster metadata and transforms to EPSG:4326.
    For GeoJSON: Calculates bounds from all features.
    """
    valid_areas_path = Path('/cephfs/work/rithvik/datasets/datasets/BHE/valid_areas/')
    for valid_path in valid_areas_path.glob("*.json"):
        if image_path.stem in valid_path.stem:
            valid_area_path = valid_path
            break
    
    # Load valid area polygon
    valid_area_gdf = None
    if valid_area_path and valid_area_path.exists():
        valid_area_gdf = gpd.read_file(valid_area_path)

    # Get valid area bounds
    if valid_area_gdf is not None:
        valid_area_bounds = valid_area_gdf.total_bounds  # (minx, miny, maxx, maxy)
        return tuple(valid_area_bounds)

    return None


def filter_false_positives_by_confidence(geojson_path, fp_confidence_threshold=0.6, output_path=None):
    """
    Filter false positives (new buildings) from baseline comparison GeoJSON based on confidence threshold.
    
    This function provides a separate post-processing step for confidence-based filtering.
    It adds a 'filtered': 'Confidence' field to low-confidence "New" buildings while keeping type='New'.
    
    This function:
    1. Loads the combined baseline comparison GeoJSON
    2. Adds 'filtered': 'Confidence' field to "New" buildings with confidence below threshold
    3. Updates colors: yellow (high conf) → purple (low conf)
    4. Updates the feature counts
    5. Saves the filtered GeoJSON
    
    Note: The baseline comparison GeoJSON must include confidence scores for "New" features.
    This is automatically done by baseline_comparison_geo() if the prediction GeoDataFrame
    contains a 'confidence' column.
    
    Feature Structure:
    - type='New', no 'filtered' field → High confidence (yellow)
    - type='New', filtered='Distance' → In buffer zone (orange)
    - type='New', filtered='Confidence' → Low confidence (purple)
    
    Parameters
    ----------
    geojson_path : str or Path
        Path to the combined baseline comparison GeoJSON file
        (output from baseline_comparison_geo)
    fp_confidence_threshold : float
        Minimum confidence threshold for keeping predictions as unfiltered "New" (0 to 1)
        Default: 0.6 (matches CONFIG in single_image_geo_analysis.py)
    output_path : str or Path, optional
        Path to save filtered GeoJSON. If None, overwrites original file.
    
    Returns
    -------
    filtered_path : str
        Path to the filtered GeoJSON file
    stats : dict
        Dictionary with filtering statistics:
        - 'original_new': Original count of unfiltered "New" buildings
        - 'filtered_new': Count of high-confidence "New" buildings (no filter)
        - 'filtered_confidence': Number of "New" buildings with filtered='Confidence'
        - 'kept_matched': Count of matched buildings (unchanged)
        - 'kept_removed': Count of removed buildings (unchanged)
        - 'kept_filtered_distance': Count of distance-filtered buildings (unchanged)
    """
    geojson_path = Path(geojson_path)
    
    # Load baseline comparison GeoJSON
    with open(geojson_path, 'r') as f:
        comparison_data = json.load(f)
    
    # Process features - reclassify low-confidence "New" as "Filtered: Confidence"
    filtered_features = []
    original_new_count = 0
    filtered_new_count = 0
    filtered_confidence_count = 0
    matched_count = 0
    removed_count = 0
    filtered_distance_count = 0
    
    for feature in comparison_data['features']:
        feature_type = feature['properties'].get('type')
        feature_filtered = feature['properties'].get('filtered')  # Check if already filtered
        
        if feature_type == 'New' and not feature_filtered:
            # Only process "New" features that haven't been filtered yet
            original_new_count += 1
            
            # Get confidence directly from feature properties
            confidence = feature['properties'].get('confidence')
            
            if confidence is not None:
                # Add filtered field based on confidence threshold
                if confidence >= fp_confidence_threshold:
                    # Keep as "New" - high confidence, no filter
                    filtered_features.append(feature)
                    filtered_new_count += 1
                else:
                    # Mark as filtered due to confidence - low confidence
                    feature['properties']['filtered'] = 'Confidence'
                    feature['properties']['color'] = '#800080'  # Purple
                    feature['properties']['description'] = 'New building filtered: low confidence'
                    filtered_features.append(feature)
                    filtered_confidence_count += 1
            else:
                # If no confidence info, keep as "New" (conservative approach)
                filtered_features.append(feature)
                filtered_new_count += 1
                pred_idx = feature['properties'].get('prediction_index', 'unknown')
                print(f"Warning: No confidence found for prediction index {pred_idx}, keeping as 'New'")
        
        elif feature_type == 'Matched':
            filtered_features.append(feature)
            matched_count += 1
        
        elif feature_type == 'Removed':
            filtered_features.append(feature)
            removed_count += 1
        
        elif feature_type == 'New' and feature_filtered == 'Distance':
            # Already filtered by distance in baseline_comparison
            filtered_features.append(feature)
            filtered_distance_count += 1
        
        elif feature_type == 'New' and feature_filtered == 'Confidence':
            # Already filtered by confidence (shouldn't happen in first pass)
            filtered_features.append(feature)
            filtered_confidence_count += 1
        
        else:
            # Unknown type, keep it
            filtered_features.append(feature)
    
    # Update feature collection with all categories
    filtered_data = {
        'type': 'FeatureCollection',
        'crs': comparison_data.get('crs', {'type': 'name', 'properties': {'name': 'EPSG:4326'}}),
        'properties': {
            'Matched': matched_count,
            'Removed': removed_count,
            'New': filtered_new_count,
            'Filtered: Distance': filtered_distance_count,
            'Filtered: Confidence': filtered_confidence_count,
            'total_features': len(filtered_features),
            'fp_confidence_threshold': fp_confidence_threshold
        },
        'features': filtered_features
    }
    
    # Save filtered GeoJSON
    if output_path is None:
        output_path = geojson_path
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    # Print statistics
    print(f"\nConfidence-Based Filtering Results:")
    print(f"  Confidence threshold: {fp_confidence_threshold}")
    print(f"  Original 'New' buildings: {original_new_count}")
    print(f"  High-confidence 'New' buildings: {filtered_new_count}")
    print(f"  Reclassified as 'Filtered: Confidence': {filtered_confidence_count}")
    print(f"  Matched buildings: {matched_count} (unchanged)")
    print(f"  Removed buildings: {removed_count} (unchanged)")
    print(f"  Filtered: Distance buildings: {filtered_distance_count} (unchanged)")
    print(f"  Saved: {output_path}")
    
    # Return statistics
    stats = {
        'original_new': original_new_count,
        'filtered_new': filtered_new_count,
        'filtered_confidence': filtered_confidence_count,
        'kept_matched': matched_count,
        'kept_removed': removed_count,
        'kept_filtered_distance': filtered_distance_count
    }
    
    return str(output_path), stats


def filter_irrelevant_building_types(geojson_path, baseline_shapefile_paths, 
                                     exclude_types=None, output_path=None,
                                     image_bounds=None):
    """
    Filter out irrelevant building types from both baseline and predictions in GeoJSON.
    
    This function identifies buildings in the baseline shapefile that are irrelevant types
    (e.g., Athletic Fields, Parks, Golf Courses) and removes them from the results:
    1. Removes from "Removed" category (they're not actual buildings to track)
    2. Removes from "New" category (if predictions match these irrelevant types)
    3. Removes from "Matched" category (correctly detected but irrelevant)
    
    Uses building ID matching for accurate filtering. Falls back to spatial intersection
    if ID fields are not available.
    
    This prevents irrelevant objects like sports fields from appearing in change detection.
    
    
    Parameters
    ----------
    geojson_path : str or Path
        Path to the combined baseline comparison GeoJSON file
    baseline_shapefile_paths : str, Path, or list
        Path(s) to baseline shapefile(s) containing building type information.
        Can be a single path or list of paths (e.g., [points.shp, polygons.shp])
    exclude_types : list or dict, optional
        Building types to exclude. Can be:
        - List of strings: ['Athletic Field', 'Golf Course', 'Park', 'Playground']
        - Dict mapping types to descriptions: {'Athletic Field': 'sports facilities', ...}
        If None, defaults to ['Athletic Field', 'Golf Course', 'Park', 'Playground']
    output_path : str or Path, optional
        Path to save filtered GeoJSON. If None, overwrites original file.
    image_bounds : tuple, optional
        Geographic bounds of the image (minx, miny, maxx, maxy) in EPSG:4326.
        If provided, only baseline buildings within these bounds are loaded and checked.
        
    
    Returns
    -------
    filtered_path : str
        Path to the filtered GeoJSON file
    stats : dict
        Dictionary with filtering statistics:
        - 'removed_baseline': Count of baseline buildings removed from 'Removed' category
        - 'removed_predictions': Count of predictions removed from 'New' category
        - 'removed_matched': Count of matched buildings removed (irrelevant but detected)
        - 'removed_types': Dict showing count per type removed
        - 'kept_matched': Count of matched buildings (after filtering)
        - 'kept_removed': Count of removed buildings (after filtering)
        - 'kept_new': Count of new buildings (after filtering)
        - 'kept_filtered_distance': Count of distance-filtered buildings (unchanged)
        - 'kept_filtered_confidence': Count of confidence-filtered buildings (unchanged)
        - 'baseline_total': Total baseline buildings in bounds (for reference)
        - 'baseline_checked': Baseline buildings actually checked (after spatial filter)
    """
    geojson_path = Path(geojson_path)
    
    # Default exclusion types
    if exclude_types is None:
        exclude_types = {
            'Athletic Field': 'sports facilities',
            'Golf Course': 'recreational facilities',
            'Park': 'public recreational spaces',
            'Playground': 'recreational areas'
        }
    
    # Convert dict to list if dictionary provided
    if isinstance(exclude_types, dict):
        types_to_exclude = list(exclude_types.keys())
        type_descriptions = exclude_types
    else:
        types_to_exclude = exclude_types
        type_descriptions = {t: 'excluded type' for t in types_to_exclude}
    
    # Load baseline shapefile(s) with spatial filtering if bounds provided
    if isinstance(baseline_shapefile_paths, list):
        # Handle multiple shapefiles
        gdfs = []
        for path in baseline_shapefile_paths:
            if image_bounds is not None:
                # Use spatial filter (bbox) for efficient loading
                gdf_part = gpd.read_file(path, bbox=image_bounds)
            else:
                gdf_part = gpd.read_file(path)
            gdfs.append(gdf_part)
        baseline_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        if baseline_gdf.crs is None and len(gdfs) > 0:
            baseline_gdf.crs = gdfs[0].crs
    else:
        if image_bounds is not None:
            # Use spatial filter (bbox) for efficient loading
            baseline_gdf = gpd.read_file(baseline_shapefile_paths, bbox=image_bounds)
        else:
            baseline_gdf = gpd.read_file(baseline_shapefile_paths)
    
    baseline_total_in_bounds = len(baseline_gdf)
    
    # Ensure CRS is EPSG:4326 for comparison with GeoJSON
    if baseline_gdf.crs != 'EPSG:4326':
        baseline_gdf = baseline_gdf.to_crs('EPSG:4326')
    
    # Find type field (case insensitive)
    type_field = None
    for field in ['Type', 'type', 'TYPE']:
        if field in baseline_gdf.columns:
            type_field = field
            break
    
    if type_field is None:
        print(f"Warning: No 'Type' field found in baseline shapefile. Available columns: {list(baseline_gdf.columns)}")
        print("Skipping building type filtering.")
        return str(geojson_path), {
            'removed_baseline': 0,
            'removed_predictions': 0,
            'removed_types': {},
            'kept_matched': 0,
            'kept_removed': 0,
            'kept_new': 0,
            'kept_filtered_distance': 0,
            'kept_filtered_confidence': 0
        }
    
    # Filter baseline to find irrelevant buildings
    irrelevant_mask = pd.Series([False] * len(baseline_gdf), index=baseline_gdf.index)
    removed_by_type = {}
    
    for exclude_type in types_to_exclude:
        type_mask = baseline_gdf[type_field].str.lower().str.contains(exclude_type.lower(), na=False)
        removed_count = type_mask.sum()
        if removed_count > 0:
            removed_by_type[exclude_type] = removed_count
            irrelevant_mask = irrelevant_mask | type_mask
    
    irrelevant_buildings = baseline_gdf[irrelevant_mask]
    
    if len(irrelevant_buildings) == 0:
        print("No irrelevant building types found in baseline. No filtering needed.")
        if image_bounds is not None:
            print(f"  (Checked {baseline_total_in_bounds} baseline buildings within image bounds)")
        return str(geojson_path), {
            'removed_baseline': 0,
            'removed_predictions': 0,
            'removed_types': {},
            'kept_matched': 0,
            'kept_removed': 0,
            'kept_new': 0,
            'kept_filtered_distance': 0,
            'kept_filtered_confidence': 0,
            'baseline_total': baseline_total_in_bounds,
            'baseline_checked': baseline_total_in_bounds
        }
    
    print(f"\nFound {len(irrelevant_buildings)} irrelevant buildings in baseline:")
    if image_bounds is not None:
        print(f"  (Out of {baseline_total_in_bounds} baseline buildings within image bounds)")
    for type_name, count in removed_by_type.items():
        description = type_descriptions.get(type_name, 'excluded type')
        print(f"  - {count} {type_name} ({description})")
    
    # Load GeoJSON
    with open(geojson_path, 'r') as f:
        comparison_data = json.load(f)
    
    # Process features
    filtered_features = []
    removed_baseline_count = 0
    removed_prediction_count = 0
    removed_matched_count = 0  # Initialize counter for filtered matched buildings
    matched_count = 0
    removed_count = 0
    new_count = 0
    filtered_distance_count = 0
    filtered_confidence_count = 0
    
    # Create a set of irrelevant building IDs for O(1) lookup instead of O(n) loop
    # This is much more efficient than iterating through irrelevant_buildings for each feature
    irrelevant_ids = set()
    id_field = None
    
    # Find the ID field in the baseline shapefile (case insensitive)
    for field in ['id', 'ID', 'Id', 'OBJECTID', 'FID', 'objectid', 'fid']:
        if field in irrelevant_buildings.columns:
            id_field = field
            irrelevant_ids = set(irrelevant_buildings[field].dropna().astype(str))
            break
    
    if id_field is None:
        print(f"Warning: No ID field found in baseline shapefile. Available columns: {list(irrelevant_buildings.columns)}")
        print("Cannot perform ID-based filtering. Consider using spatial filtering instead.")
    else:
        print(f"Using ID field '{id_field}' for filtering ({len(irrelevant_ids)} irrelevant building IDs)")
    
    for feature in comparison_data['features']:
        feature_type = feature['properties'].get('type')
        feature_filtered = feature['properties'].get('filtered')
        
        # Get building ID from feature properties (check multiple possible field names)
        feature_id = None
        for id_key in ['building_id', 'id', 'baseline_id', 'truth_id', 'ID']:
            if id_key in feature['properties']:
                feature_id = feature['properties'].get(id_key)
                if feature_id is not None:
                    feature_id = str(feature_id)  # Ensure string for comparison
                    break
        
        should_remove = False
        
        # Check if this feature's ID matches any irrelevant baseline building
        # Only check if we have both a valid ID field and a feature ID
        if id_field is not None and feature_id is not None and feature_id in irrelevant_ids:
            should_remove = True
                
            if feature_type == 'Removed':
                removed_baseline_count += 1
            elif feature_type == 'New':
                removed_prediction_count += 1
            elif feature_type == 'Matched':
                removed_matched_count += 1
        
        # Keep or remove feature based on analysis
        if should_remove:
            # Skip this feature (don't add to filtered_features)
            continue
        else:
            # Keep the feature
            filtered_features.append(feature)
            
            # Count by category
            if feature_type == 'Matched':
                matched_count += 1
            elif feature_type == 'Removed':
                removed_count += 1
            elif feature_type == 'New':
                if feature_filtered == 'Distance':
                    filtered_distance_count += 1
                elif feature_filtered == 'Confidence':
                    filtered_confidence_count += 1
                else:
                    new_count += 1
    
    # Update feature collection
    filtered_data = {
        'type': 'FeatureCollection',
        'crs': comparison_data.get('crs', {'type': 'name', 'properties': {'name': 'EPSG:4326'}}),
        'properties': {
            'Matched': matched_count,
            'Removed': removed_count,
            'New': new_count,
            'Filtered: Distance': filtered_distance_count,
            'Filtered: Confidence': filtered_confidence_count,
            'total_features': len(filtered_features),
            'building_type_filtering_applied': True,
            'excluded_types': types_to_exclude
        },
        'features': filtered_features
    }
    
    # Save filtered GeoJSON
    if output_path is None:
        output_path = geojson_path
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    # Print statistics
    print(f"\nBuilding Type Filtering Results:")
    print(f"  Removed from baseline ('Removed' category): {removed_baseline_count}")
    print(f"  Removed from predictions ('New' category): {removed_prediction_count}")
    print(f"  Removed from 'Matched' category: {removed_matched_count}")
    print(f"  Kept Matched: {matched_count}")
    print(f"  Kept Removed (after filtering): {removed_count}")
    print(f"  Kept New (after filtering): {new_count}")
    print(f"  Kept Filtered: Distance: {filtered_distance_count}")
    print(f"  Kept Filtered: Confidence: {filtered_confidence_count}")
    print(f"  Total features: {len(filtered_features)}")
    print(f"  Saved: {output_path}")
    
    # Return statistics
    stats = {
        'removed_baseline': removed_baseline_count,
        'removed_predictions': removed_prediction_count,
        'removed_matched': removed_matched_count,
        'removed_types': removed_by_type,
        'kept_matched': matched_count,
        'kept_removed': removed_count,
        'kept_new': new_count,
        'kept_filtered_distance': filtered_distance_count,
        'kept_filtered_confidence': filtered_confidence_count,
        'baseline_total': baseline_total_in_bounds,
        'baseline_checked': baseline_total_in_bounds
    }
    
    return str(output_path), stats
