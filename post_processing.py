"""
Post-processing functions for GeoJSON-based building detection analysis.

This module provides functions for filtering and processing GeoJSON results
from baseline comparison, including confidence-based false positive filtering.
"""

import geopandas as gpd
import json
from pathlib import Path


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
