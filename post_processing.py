"""
Post-processing functions for GeoJSON-based building detection analysis.

This module provides functions for filtering and processing GeoJSON results
from baseline comparison, including confidence-based false positive filtering.
"""

import geopandas as gpd
import json
from pathlib import Path


def filter_false_positives_by_confidence(geojson_path, fp_confidence_threshold=0.79, output_path=None):
    """
    Filter false positives (new buildings) from baseline comparison GeoJSON based on confidence threshold.
    
    This function:
    1. Loads the combined baseline comparison GeoJSON
    2. Filters out "New" buildings with confidence below threshold
    3. Updates the feature counts
    4. Saves the filtered GeoJSON
    
    Note: The baseline comparison GeoJSON must include confidence scores for "New" features.
    This is automatically done by baseline_comparison_geo() if the prediction GeoDataFrame
    contains a 'confidence' column.
    
    Parameters
    ----------
    geojson_path : str or Path
        Path to the combined baseline comparison GeoJSON file
        (output from baseline_comparison_geo)
    fp_confidence_threshold : float
        Minimum confidence threshold for keeping false positives (0 to 1)
        Default: 0.79 (same as post_processing_tools.py)
    output_path : str or Path, optional
        Path to save filtered GeoJSON. If None, overwrites original file.
    
    Returns
    -------
    filtered_path : str
        Path to the filtered GeoJSON file
    stats : dict
        Dictionary with filtering statistics:
        - 'original_new': Original count of new buildings
        - 'filtered_new': Count of new buildings after filtering
        - 'removed_count': Number of new buildings removed
        - 'kept_matched': Count of matched buildings (unchanged)
        - 'kept_removed': Count of removed buildings (unchanged)
    """
    geojson_path = Path(geojson_path)
    
    # Load baseline comparison GeoJSON
    with open(geojson_path, 'r') as f:
        comparison_data = json.load(f)
    
    # Filter features
    filtered_features = []
    original_new_count = 0
    filtered_new_count = 0
    matched_count = 0
    removed_count = 0
    removed_fp_count = 0
    
    for feature in comparison_data['features']:
        feature_type = feature['properties'].get('type')
        
        if feature_type == 'New':
            original_new_count += 1
            
            # Get confidence directly from feature properties
            confidence = feature['properties'].get('confidence')
            
            if confidence is not None:
                # Keep only if confidence meets threshold
                if confidence >= fp_confidence_threshold:
                    filtered_features.append(feature)
                    filtered_new_count += 1
                else:
                    removed_fp_count += 1
            else:
                # If no confidence info, keep the feature (conservative approach)
                filtered_features.append(feature)
                filtered_new_count += 1
                pred_idx = feature['properties'].get('prediction_index', 'unknown')
                print(f"Warning: No confidence found for prediction index {pred_idx}, keeping feature")
        
        elif feature_type == 'Matched':
            filtered_features.append(feature)
            matched_count += 1
        
        elif feature_type == 'Removed':
            filtered_features.append(feature)
            removed_count += 1
        
        else:
            # Unknown type, keep it
            filtered_features.append(feature)
    
    # Update feature collection
    filtered_data = {
        'type': 'FeatureCollection',
        'crs': comparison_data.get('crs', {'type': 'name', 'properties': {'name': 'EPSG:4326'}}),
        'properties': {
            'Matched': matched_count,
            'Removed': removed_count,
            'New': filtered_new_count,
            'total_features': len(filtered_features),
            'fp_confidence_threshold': fp_confidence_threshold,
            'filtered_false_positives': removed_fp_count
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
    print(f"\nFalse Positive Filtering Results:")
    print(f"  Confidence threshold: {fp_confidence_threshold}")
    print(f"  Original new buildings: {original_new_count}")
    print(f"  Filtered new buildings: {filtered_new_count}")
    print(f"  Removed (low confidence): {removed_fp_count}")
    print(f"  Matched buildings: {matched_count} (unchanged)")
    print(f"  Removed buildings: {removed_count} (unchanged)")
    print(f"  Saved: {output_path}")
    
    # Return statistics
    stats = {
        'original_new': original_new_count,
        'filtered_new': filtered_new_count,
        'removed_count': removed_fp_count,
        'kept_matched': matched_count,
        'kept_removed': removed_count
    }
    
    return str(output_path), stats
