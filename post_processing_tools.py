"""
This script is designed to perform the following tasks:
1. Provide functions for loading models and images
2. Generate predictions using YOLO models
3. Save and load predictions
4. Provide helper functions for post processing and filtering
5. Perform analysis on predictions and ground truth
6. Visualize results and save annotated images
"""
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import cv2
import rasterio
from shapely.geometry import Point, box as shp_box
import pyproj
from shapely.ops import transform as shapely_transform, nearest_points
from tqdm import tqdm
from sys import path
from evaluation import (create_final_histogram, create_final_pie_chart, 
                  calculate_box_metrics, non_max_suppression)


# --- LOADING MODELS IMAGES AND PREDICTIONS ---

def load_models(model_type, model_version):
    """
    Load models based on type and version usually YOLO kfolds
    
    Parameters
    ----------
    model_type : str
        Type of model to load ('yolo', 'kfolds', or 'rcnn')
    model_version : str
        Version of the model to load (for YOLO or kfolds)

    Returns
    ----------
    models : list
        List of loaded model(s)
        
    """
    if model_type == 'yolo':
        model = YOLO("./OE_CL_Processing/pre_trained/weights/best.pt") #change to correct directory
        return [model]  # Return as list for consistent handling
    elif model_type == 'kfolds':
        k_fold_models = []
        model_dir = Path(f"./OE_CL_Processing/k_folds_cross_val_{model_version}") #change to correct directory

        for k in range(5):
            model_path = model_dir / f"split_{k+1}" / "weights" / "best.pt"
            if model_path.exists():
                k_fold_models.append(YOLO(model_path))
        return k_fold_models
    else:  # rcnn
        return None  # RCNN doesn't need model loading in the same way
    
    
def get_image_paths(image_dir):
    """
    Get list of image paths
    
    Parameters
    ----------
    image_dir : str
        Directory containing images
    
    Returns
    ----------
    image_files : list
        List of image file paths
    """
    image_files = []
    for ext in ['.png', '.tif', '.tiff']:
        image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
    print(f"Found {len(image_files)} images")
    return sorted(image_files)  # Sort for consistent ordering


def generate_predictions(models, image_files, output_dir, model_type, conf_threshold):
    """
    Generate predictions for all images and save them
    
    Parameters
    -----------
    models : list
        List of loaded models for prediction
    image_files : list or Path
        List of image file paths or single image path
    output_dir : str
        Directory to save predictions
    model_type : str
        Type of model ('yolo', 'kfolds', or 'rcnn')
    conf_threshold : float
        Confidence threshold for predictions
    Returns
    --------
    all_predictions : dict
        Dictionary of bbox predictions for each image
    all_confidences : dict
        Dictionary of confidence scores for each bbox for each image
    """
    
    if not isinstance(image_files, list): # Ensure image_files is a list
        image_files = [image_files]
    
    all_predictions = {}
    all_confidences = {}
    prediction_dir = Path(output_dir) / "predictions"
    prediction_dir.mkdir(exist_ok=True, parents=True)
    
    for img_path in tqdm(image_files, desc="Generating predictions"):
        pred_path = prediction_dir / f"{img_path.stem}_pred.npy"
        conf_path = prediction_dir / f"{img_path.stem}_conf.npy"
        
        # Skip if predictions already exist
        if pred_path.exists() and conf_path.exists():
            all_predictions[str(img_path)] = np.load(pred_path)
            all_confidences[str(img_path)] = np.load(conf_path)
            continue
            
        # Get predictions based on model type
        if model_type == 'rcnn':
            rcnn_pred_path = Path('path/to/rcnn/outputs') / f"{img_path.stem}.txt" #change to correct directory if it exists
            pred_boxes = load_boxes(rcnn_pred_path)
            pred_confidences = [1.0] * len(pred_boxes)  # Default confidence for RCNN
        else:  # yolo or kfolds
            all_boxes = []
            all_confidences_raw = []
            
            for model in models:
                results = model.predict([str(img_path)], conf=conf_threshold, verbose=False)
                for result in results:
                    all_boxes.extend(result.boxes.xyxy.cpu().numpy())
                    all_confidences_raw.extend(result.boxes.conf.cpu().numpy())
            
            # Apply NMS to consolidate predictions
            pred_boxes, pred_confidences = non_max_suppression(all_boxes, all_confidences_raw)
        
        # Save predictions
        all_predictions[str(img_path)] = pred_boxes
        all_confidences[str(img_path)] = pred_confidences
        np.save(pred_path, pred_boxes)
        np.save(conf_path, pred_confidences)
        
    return all_predictions, all_confidences

def load_saved_predictions(image_files, output_dir):
    """
    Load previously saved predictions if they are not generated by sliding window
    
    Parameters
    ----------
    image_files : list
        List of image file paths
    output_dir : str
        Directory where predictions are saved
    Returns
    -------
    all_predictions : dict
        Dictionary with image paths as keys and prediction boxes as values
    """
    all_predictions = {}
    prediction_dir = Path(output_dir) / "predictions"
    
    for img_path in image_files:
        pred_path = prediction_dir / f"{img_path.stem}_pred.npy"
        if pred_path.exists():
            pred_boxes = np.load(pred_path)
            all_predictions[str(img_path)] = pred_boxes
        else:
            print(f"Warning: No saved predictions for {img_path.stem}")
    
    print(f"Loaded predictions for {len(all_predictions)} images")
    return all_predictions

# --- FILTERING PREDICTIONS---
def filter_buildings_by_pipeline_distance(pred_boxes, image_path, pipeline_path, max_distance_meters=100):
    """
    Filter predicted buildings based on their distance to a pipeline
    
    Parameters
    -----------
    pred_boxes : list of lists
        Predicted building boxes in pixel coordinates [x1, y1, x2, y2]
    image_path : str
        Path to the GeoTIFF image
    pipeline_path : str
        Path to the pipeline shapefile
    max_distance_meters : float
        Maximum allowed distance in meters from pipeline
        
    Returns
    --------
    filtered_boxes : list of lists
        Boxes that are within the distance threshold
    rejected_boxes : list of lists
        Boxes that are too far from the pipeline
    distances : list of float
        Distance of each filtered box to the nearest pipeline segment
    """
    # Load pipeline shapefile
    pipeline_gdf = gpd.read_file(pipeline_path)
    
    # Ensure the shapefile is loaded successfully
    if pipeline_gdf.empty:
        print(f"Warning: Pipeline shapefile is empty or failed to load: {pipeline_path}")
        return pred_boxes, [], []
        
    # Check if image is GeoTIFF
    if not str(image_path).lower().endswith(('.tif', '.tiff')):
        print(f"Warning: Image is not a GeoTIFF, cannot perform geographic filtering: {image_path}")
        return pred_boxes, [], []
    
    # Get geotransform from GeoTIFF
    with rasterio.open(image_path) as src:
        transform = src.transform
        crs = src.crs
    
    # Ensure pipeline CRS matches image CRS
    if pipeline_gdf.crs != crs:
        pipeline_gdf = pipeline_gdf.to_crs(crs)
        
    # Convert building boxes to geographic coordinates
    geo_boxes = []
    building_geoms = []
    
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Transform the four corners to geographic coordinates
        ul_x, ul_y = transform * (x1, y1)  # Upper left
        ur_x, ur_y = transform * (x2, y1)  # Upper right (not needed for YOLO)
        lr_x, lr_y = transform * (x2, y2)  # Lower right
        ll_x, ll_y = transform * (x1, y2)  # Lower left (not needed for YOLO)
        
        # Create a polygon geometry representing the building footprint
        building_poly = shp_box(ul_x, ul_y, lr_x, lr_y)
        building_geoms.append(building_poly)
    # Get the pipeline geometry (might be MultiLineString)
    pipeline_geom = pipeline_gdf.geometry.union_all('unary')
    
    # Create buffer for containment testing - more consistent than distance checks
    try:
        # Check CRS units
        is_projected = crs.is_projected
        
        if is_projected:
            # For projected CRS (like UTM), use regular buffer
            buffer_geom = pipeline_geom.buffer(max_distance_meters)
        else:
            # For geographic CRS (like WGS84), create a geodesic buffer
            # Get center point of pipeline to determine UTM zone
            pipeline_centroid = pipeline_geom.centroid
            
            # Find appropriate UTM zone for this location
            utm_band = int((pipeline_centroid.x + 180) / 6) + 1
            utm_epsg = 32600 + utm_band if pipeline_centroid.y >= 0 else 32700 + utm_band
            utm_crs = pyproj.CRS.from_epsg(utm_epsg)
            
            # Create the transformer functions
            project_to_utm = pyproj.Transformer.from_crs(
                crs, utm_crs, always_xy=True).transform
            project_to_orig = pyproj.Transformer.from_crs(
                utm_crs, crs, always_xy=True).transform
            
            # Convert pipeline to UTM, buffer, then back to original CRS
            pipeline_utm = shapely_transform(project_to_utm, pipeline_geom)
            buffer_utm = pipeline_utm.buffer(max_distance_meters)
            buffer_geom = shapely_transform(project_to_orig, buffer_utm)
    except Exception as e:
        # Fallback to approximate degree buffer if geodesic method fails
        print(f"Warning: Couldn't create proper buffer: {e}")
        print("Using approximate distance calculation instead")
        buffer_geom = None  # Will fall back to distance calculation
    
    # Calculate distance of each building to the pipeline
    distances = []
    filtered_indices = []
    rejected_indices = []
    
    for i, (box, geom) in enumerate(zip(pred_boxes, building_geoms)):
        # Use the helper function for accurate distance calculation in meters
        dist = calculate_distance(geom.centroid, pipeline_geom, crs)
        
        # Check if building polygon intersects with the buffer
        if buffer_geom is not None and buffer_geom.intersects(geom):
            filtered_indices.append(i)
            distances.append(dist)
            # print (f"Building {i} is within buffer")
        # Fall back to distance check if buffer fails
        elif dist <= max_distance_meters:
            filtered_indices.append(i)
            distances.append(dist)
            # print(f"Building {i} is within distance threshold")
        else:
            rejected_indices.append(i)
    
    # Filter boxes
    filtered_boxes = [pred_boxes[i] for i in filtered_indices]
    rejected_boxes = [pred_boxes[i] for i in rejected_indices]
    
    return filtered_boxes, rejected_boxes, distances

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
            intersection = calculate_iou(box1, box2) 
            smaller_area = min(area1, area2)
            overlap_ratio = intersection * (area1 + area2) / smaller_area
            
            if overlap_ratio >= overlap_threshold:
                if idx1 not in filtered_indices:
                    filtered_indices.add(idx1) 
    
    return filtered_indices

def load_boxes(filepath):
    """
    Load bounding boxes from a file.
    
    Parameters
    --------
    filepath : str
        Path to the file containing box coordinates
    
    Returns
    --------
    boxes : Numpy array
        Array of bounding boxes. Each box is represented as [x1, y1, x2, y2] or
        [x_center, y_center, width, height] depending on the format.
    """
    boxes = []
    with open(filepath, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            
            if len(values) == 5:
                # YOLO format: class_id, x_center, y_center, width, height
                # Just store the raw values (excluding class_id)
                boxes.append(values[1:])  # Store as [x_center, y_center, width, height]
            elif len(values) == 4:
                # Standard format: x1, y1, x2, y2
                boxes.append(values)
            else:
                print(f"Warning: Invalid line format in {filepath}: {line.strip()}")
    
    return np.array(boxes)

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Parameters
    ----------
    box1, box2 : list
        List of coordinates [x1, y1, x2, y2]
    
    Returns
    -------
    iou : float
        IoU value between box1 and box2
    """
    # Convert to shapely boxes
    b1 = shp_box(box1[0], box1[1], box1[2], box1[3])
    b2 = shp_box(box2[0], box2[1], box2[2], box2[3])
    
    if not b1.intersects(b2):
        return 0.0
    
    intersection = b1.intersection(b2).area
    union = b1.area + b2.area - intersection
    return intersection / union



# --- POST PROCESSING AND VISUALIZATION ---

def post_processing_analysis(pred_path, truth_path, image_path, output_dir, pipeline_shp_path=None, found_shp_file=None, 
                               max_distance=100, truth_is_YOLO=True, point_distance_tolerance=10, 
                               pred_confidences=None, fp_confidence=0.79, testing=False, save_images=True):
    """
    Analyzes building detections, comparing predictions against ground truth with special handling for points and boxes.
    
    Parameters
    ----------
    pred_path: str
        Path to prediction file
    truth_path: str
        Path to ground truth file
    image_path: str
        Path to image file
    output_dir: str
        Directory to save output
    pipeline_shp_path: Optional [str]
        Path to pipeline shapefile
    found_shp_file: Optional [str]
        Path to reported buildings shapefile
    max_distance: float
        Maximum distance in meters for pipeline buffer
    truth_is_YOLO: bool
        Whether truth boxes are in YOLO format (legacy parameter can be removed since it will always be true)
    point_distance_tolerance: float
        Distance tolerance for point matching in meters
    pred_confidences: Optional [List[float]]
        List of prediction confidence scores for false positive filtering
    fp_confidence: float
        Confidence threshold for filtering false positives (0 to 1)
    testing: bool
        Whether to run additional testing for comparison with baseline shapefiles (can possibly be removed since it will always be true)
    save_images: bool
        Whether to save annotated images (False to skip for space efficiency)

    Returns
    -------
    missed: list
        List of missed building boxes in normalized pixel coordinates [x1, y1, x2, y2]
    false_positive: list
        List of false positive building boxes in normalized pixel coordinates [x1, y1, x2, y2]
    output_path: str
        Path to the saved annotated image
    fp_confidences: list
        List of confidence scores for false positive boxes
    found_matches_info: dict
        Metrics about found building matches to be used for analysis
    all_testing_buildings_filtered: GeoDataFrame
        Filtered GeoDataFrame of all reported buildings for the current image
    shapefile_objects_actually_matched_by_model_count: int
        Count of reported shapefile objects actually matched by model predictions
    count_new_sf_buildings: int
        Count of shapefile objects missed by the model (new and removed)    
    """
    # --- 1. INITIALIZATION AND IMAGE LOADING ---
    image, height, width, transform, crs, is_geotiff, bounds_geo = load_and_prepare_image(image_path)
    annotated_image = image.copy() if save_images else None
    
    # --- 2. LOAD AND PREPARE BOXES ---
    pred_boxes = load_and_filter_predictions(pred_path, overlap_threshold=0.7)
    truth_boxes = load_truth_boxes(truth_path, truth_is_YOLO, width, height)
    
    # --- 3. PREPARE ADDITIONAL DATA FOR TESTING ---
    removed_buildings, found_buildings, removed_box_matches, found_box_matches = None, None, {}, {}
    inverse_transform = None
    all_testing_buildings_filtered = gpd.GeoDataFrame() # Initialize as empty GeoDataFrame
    
    if testing and is_geotiff:
        # Load shapefile data for found and removed buildings
        found_buildings, removed_buildings = load_shapefiles_for_testing(crs, found_shp_file=found_shp_file)
        inverse_transform = ~transform if transform else None
        
        all_testing_buildings = None 
        
        gdfs_to_concat = []
        if found_buildings is not None and not found_buildings.empty:
            gdfs_to_concat.append(found_buildings)
        if removed_buildings is not None and not removed_buildings.empty:
            gdfs_to_concat.append(removed_buildings)
    
        
        # Concatenate if there's anything to concatenate
        if gdfs_to_concat:
            all_testing_buildings = pd.concat(gdfs_to_concat, ignore_index=True)
    
        if all_testing_buildings is not None and not all_testing_buildings.empty and bounds_geo is not None:
                # Create a polygon representing the image bounds
                image_bounds_polygon = shp_box(bounds_geo[0], bounds_geo[1], bounds_geo[2], bounds_geo[3])
                
                # Filter all_testing_buildings to include only polygons that intersect with the image bounds
                # Ensure all_testing_buildings is a GeoDataFrame and has a 'geometry' column
                if isinstance(all_testing_buildings, gpd.GeoDataFrame) and 'geometry' in all_testing_buildings.columns:
                    all_testing_buildings_filtered = all_testing_buildings[all_testing_buildings.geometry.intersects(image_bounds_polygon)]
                else:
                    print("Warning: all_testing_buildings is not a valid GeoDataFrame or is missing a geometry column.")
                    # all_testing_buildings_filtered is already initialized to empty
        # else: # all_testing_buildings_filtered is already initialized if no data or no bounds
            # all_testing_buildings_filtered = gpd.GeoDataFrame() 

            
        
        # Initialize tracking dictionaries
        found_box_matches = {"image_id": Path(image_path).stem, "matching_boxes": []}
        removed_box_matches = {"image_id": Path(image_path).stem, "matching_boxes": []}
    
  
    
    # --- 4. PIPELINE VISUALIZATION AND FILTERING ---
    if pipeline_shp_path and is_geotiff:
        pred_boxes, rejected_boxes, distances = filter_buildings_by_pipeline_distance(
            pred_boxes, image_path, pipeline_shp_path, max_distance_meters=max_distance
        )
        
        # Draw rejected boxes in gray
        if save_images:
            for box in rejected_boxes:
                cv2.rectangle(annotated_image, (int(box[0]), int(box[1])), 
                             (int(box[2]), int(box[3])), (128, 128, 128), 1)
        
        # Visualize pipeline and buffer zone
        if is_geotiff and save_images:
            annotated_image = visualize_pipeline_and_buffer(
                annotated_image, pipeline_shp_path, crs, transform, 
                inverse_transform, bounds_geo, height, width, max_distance
            )
    
    # --- 5. CLASSIFICATION OF TRUTH BOXES ---
    point_indices, box_indices = classify_truth_boxes(truth_boxes)
    
    # --- 6. MATCHING PROCESS ---
    matched_predictions, matched_truths = set(), set()
    
    # Phase 1: Find and process box-point pairs
    box_point_pairs, point_indices = find_box_point_pairs(truth_boxes, point_indices, box_indices)
    
    # Phase 2: Process box-point pairs
    processed_pairs = match_box_point_pairs(
        box_point_pairs, truth_boxes, pred_boxes, 
        matched_truths, matched_predictions, 
        annotated_image if save_images else None
    )
    
    # Phase 3: Match standalone boxes
    match_standalone_boxes(
        box_indices, truth_boxes, pred_boxes,
        matched_truths, matched_predictions, 
        annotated_image if save_images else None
    )
    
    # Phase 4: Match standalone points
    point_matches = find_potential_point_matches(
        point_indices, truth_boxes, pred_boxes,
        processed_pairs, matched_truths, matched_predictions, point_distance_tolerance,
        pred_confidences=pred_confidences  # Pass pred_confidences
    )
    
    # Assign optimal point matches
    assign_point_matches(
        point_matches, truth_boxes, pred_boxes,
        matched_truths, matched_predictions, 
        annotated_image if save_images else None
    )
    
    # --- 7. IDENTIFY MISSED DETECTIONS ---
    missed_points = [i for i in point_indices if i not in matched_truths]
    missed_boxes = [i for i in box_indices if i not in matched_truths]
    missed = [truth_boxes[i] for i in (missed_points + missed_boxes)]
    
    # --- 8. PROCESS REMOVED BUILDINGS SHAPEFILE ---
    if save_images and testing and is_geotiff and removed_buildings is not None:
        matching_indices = process_removed_buildings(
            missed, removed_buildings, transform, inverse_transform,
            bounds_geo, pred_boxes, matched_predictions, annotated_image
        )
        removed_box_matches["matching_boxes"] = matching_indices
        
        # Add legends and write to file if matches found
        if matching_indices:
            add_removed_buildings_legend(annotated_image, output_dir, removed_box_matches)
    elif testing and is_geotiff and removed_buildings is not None:
        # Just calculate matching indices without visualizing
        matching_indices = identify_removed_building_matches(
            missed, removed_buildings, transform
        )
        removed_box_matches["matching_boxes"] = matching_indices
    
    # --- 9. MARK MISSED BOXES ---
    if save_images:
        for box in missed:
            cv2.rectangle(annotated_image, (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), (0, 0, 255), 2)
    
    # --- 10. PROCESS FALSE POSITIVES ---
    false_positive, fp_confidences = get_filtered_false_positives(
        pred_boxes, matched_predictions, pred_confidences, fp_confidence=fp_confidence
    )
    
    # --- 11. PROCESS FOUND BUILDINGS SHAPEFILE ---
    if save_images and testing and is_geotiff and found_buildings is not None:
        matching_indices = process_found_buildings(
            false_positive, found_buildings, transform, inverse_transform,
            bounds_geo, pred_boxes, annotated_image
        )
        found_box_matches["matching_boxes"] = matching_indices
        
        # Add legends and write to file if matches found
        if matching_indices:
            add_found_buildings_legend(annotated_image, output_dir, found_box_matches)
    elif testing and is_geotiff and found_buildings is not None:
        # Just calculate matching indices without visualizing
        matching_indices = identify_found_building_matches(
            false_positive, found_buildings, transform, pred_boxes
        )
        found_box_matches["matching_boxes"] = matching_indices
    
    # ---  COUNT SHAPEFILE OBJECTS ACTUALLY MATCHED BY MODEL PREDICTIONS ---
    shapefile_objects_actually_matched_by_model_count = 0
    pred_polygons_for_check = []
    if testing and is_geotiff and not all_testing_buildings_filtered.empty and transform is not None:
        if pred_boxes is not None and len(pred_boxes) > 0:
            geo_pred_polygons = []
            for box_px in pred_boxes:
                x1, y1, x2, y2 = map(int, box_px)
                ul_x, ul_y = transform * (x1, y1)
                lr_x, lr_y = transform * (x2, y2)
                # Ensure correct ordering for shp_box if not already minx, miny, maxx, maxy
                min_x, max_x = min(ul_x, lr_x), max(ul_x, lr_x)
                min_y, max_y = min(ul_y, lr_y), max(ul_y, lr_y)
                pred_poly = shp_box(min_x, min_y, max_x, max_y)
                geo_pred_polygons.append(pred_poly)

            for sf_geom in all_testing_buildings_filtered.geometry:
                # if not sf_geom.is_valid:
                #     continue
                for pred_geom in geo_pred_polygons:
                    if not pred_geom.is_valid:
                        continue
                    if sf_geom.intersects(pred_geom):
                        try:
                            intersection = sf_geom.intersection(pred_geom)
                            if not intersection.is_empty and intersection.area > 1e-9: # Threshold for meaningful overlap
                                shapefile_objects_actually_matched_by_model_count += 1
                                break # Count this shapefile object once
                        except Exception:
                            pass 
    # --- IDENTIFY NEW/REMOVED SHAPEFILE BUILDINGS MISSED BY THE MODEL ---
    count_new_sf_buildings = 0
    if testing and is_geotiff and not all_testing_buildings_filtered.empty and transform is not None:
        if not all_testing_buildings_filtered.empty:
            for sf_geom in all_testing_buildings_filtered.geometry:
                # if not sf_geom.isvalid:
                #     continue

                reported_sf_check = False
                for pred_geom in pred_polygons_for_check:
                    # if not pred_geom.is_valid:
                    #     continue
                    if sf_geom.intersects(pred_geom):
                        try:
                            intersection = sf_geom.intersection(pred_geom)
                            if not intersection.is_empty and intersection.area > 1e-9: # Threshold for meaningful overlap
                                reported_sf_check = True
                                break # Count this shapefile object once
                        except Exception:
                            pass
                if not reported_sf_check:
                    count_new_sf_buildings += 1
                    
                    


    # --- 12. MARK FALSE POSITIVES ---
    if save_images:
        mark_false_positives(
            false_positive, fp_confidences, pred_boxes, annotated_image,
            found_box_matches.get("matching_boxes", []) if testing else []
        )
    
    # --- 13. SAVE AND RETURN RESULTS ---
    output_path = ""
    if save_images:
        output_path = str(Path(output_dir) / f"{Path(image_path).stem}_analysis.jpg")
        cv2.imwrite(output_path, annotated_image)
    
    # Return relevant information
    found_matches_info = found_box_matches if testing and "matching_boxes" in found_box_matches and found_box_matches["matching_boxes"] else None
    
    return missed, false_positive, output_path, fp_confidences, found_matches_info, all_testing_buildings_filtered, shapefile_objects_actually_matched_by_model_count, count_new_sf_buildings


# --- HELPER FUNCTIONS ---

def identify_removed_building_matches(missed, removed_buildings, transform):
    """
    Calculate removed building matches without visualization
    
    Parameters
    -----------
    missed : list of lists
        List of missed building boxes in pixel coordinates [x1, y1, x2, y2]
    removed_buildings : GeoDataFrame
        GeoDataFrame of removed buildings
    transform : pyproj.Transformer
        Transformer for converting pixel coordinates to geographic coordinates
    Returns
    --------
    matching_indices : list of int
        List of indices of missed boxes that match removed buildings
    """
    # Create building geometries from missed buildings only
    missed_geoms = []
    for box in missed:  
        x1, y1, x2, y2 = map(int, box)
        
        # Transform to geographic coordinates
        ul_x, ul_y = transform * (x1, y1)
        lr_x, lr_y = transform * (x2, y2)
        
        # Create polygon for building footprint
        missed_poly = shp_box(ul_x, ul_y, lr_x, lr_y)
        missed_geoms.append(missed_poly)
    
    # Check each building geometry for intersection with removed buildings
    removed_matches = []
    matched_removed_buildings = set()
    
    for i, geom in enumerate(missed_geoms):
        for removed_idx, removed_geom in enumerate(removed_buildings.geometry):
            if geom.is_valid and removed_geom.is_valid and geom.intersects(removed_geom):
                try:
                    intersection = geom.intersection(removed_geom)
                    if not intersection.is_empty:
                        intersection_area = intersection.area
                        if intersection_area > 1e-9:
                            removed_matches.append((i, removed_idx, intersection_area))
                            matched_removed_buildings.add(removed_idx)
                except Exception:
                    pass
    
    # Find best matches based on intersection area
    best_match_for_removed = {}
    for miss_idx, removed_idx, area in removed_matches:
        if removed_idx not in best_match_for_removed or area > best_match_for_removed[removed_idx][1]:
            best_match_for_removed[removed_idx] = (miss_idx, area)
    
    # Get unique missed box indices that are the best match
    missed_indices_to_mark = {match[0] for match in best_match_for_removed.values()}
    
    # Create list of matching indices
    matching_indices = [idx for idx in missed_indices_to_mark if 0 <= idx < len(missed)]
    
    return matching_indices

def identify_found_building_matches(filtered_fp, found_buildings, transform, pred_boxes):
    """
    Calculate found building matches without visualization

    Parameters
    -----------
    filtered_fp : list of lists
        List of filtered false positive building boxes in pixel coordinates [x1, y1, x2, y2]
    found_buildings : GeoDataFrame
        GeoDataFrame of found buildings
    transform : pyproj.Transformer
        Transformer for converting pixel coordinates to geographic coordinates 
    pred_boxes : list of lists
        List of predicted building boxes in pixel coordinates [x1, y1, x2, y2]
    
    Returns
    --------
    matching_indices : list of int
        List of indices of filtered false positive boxes that match found buildings
    """
    # Create building geometries from false positives
    building_geoms = []
    for box in filtered_fp:  
        x1, y1, x2, y2 = map(int, box)
        
        # Transform to geographic coordinates
        ul_x, ul_y = transform * (x1, y1)
        lr_x, lr_y = transform * (x2, y2)
        
        # Create polygon for building footprint
        building_poly = shp_box(ul_x, ul_y, lr_x, lr_y)
        building_geoms.append(building_poly)
    
    # Find intersections between false positives and found buildings
    found_matches = []
    
    for i, geom in enumerate(building_geoms):
        for found_idx, found_geom in enumerate(found_buildings.geometry):
            if geom.is_valid and found_geom.is_valid and geom.intersects(found_geom):
                try:
                    intersection = geom.intersection(found_geom)
                    if not intersection.is_empty:
                        intersection_area = intersection.area
                        if intersection_area > 1e-9:
                            found_matches.append((i, found_idx, intersection_area))
                except Exception:
                    pass
    
    # Find best matches based on intersection area
    best_match_for_found = {}
    for fp_idx, found_idx, area in found_matches:
        if found_idx not in best_match_for_found or area > best_match_for_found[found_idx][1]:
            best_match_for_found[found_idx] = (fp_idx, area)
    
    # Get unique false positive indices that are best matches
    fp_indices_to_mark = {match[0] for match in best_match_for_found.values()}
    
    # Map back to original prediction indices
    matching_indices = []
    for fp_idx in fp_indices_to_mark:
        if 0 <= fp_idx < len(filtered_fp):
            box = filtered_fp[fp_idx]
            # Find original index in pred_boxes
            orig_indices = [j for j, pb in enumerate(pred_boxes) if np.array_equal(pb, box)]
            if orig_indices:
                matching_indices.append(orig_indices[0])
                
    return matching_indices

def load_and_prepare_image(image_path):
    """
    Loads image and extracts relevant metadata
    
    Parameters
    ----------
    image_path : str
        Path to the image file
    
    Returns
    -------
    image : numpy array
        Loaded image
    height : int
        Height of the image in pixels
    width : int
        Width of the image in pixels
    transform : pyproj.Transformer
        Transformer for converting pixel coordinates to geographic coordinates
    crs : pyproj.CRS
        Coordinate Reference System of the image
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
            height, width = image.shape[:2]
            crs = src.crs
            bounds_geo = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        height, width = image.shape[:2]
        transform, crs, bounds_geo = None, None, None
    
    return image, height, width, transform, crs, is_geotiff, bounds_geo

def load_and_filter_predictions(pred_path, overlap_threshold=0.7):
    """
    Loads prediction boxes and filters overlapping ones
    
    Parameters
    ----------
    pred_path : str
        Path to the prediction file
    overlap_threshold : float
        Threshold for filtering overlapping boxes (0 to 1)
    
    Returns
    -------
    filtered_pred_boxes : list of lists
        List of filtered predicted bounding boxes in the format [x1, y1, x2, y2]
    """
    pred_boxes = load_boxes(pred_path)
    filtered_pred_indices = filter_overlapping_boxes(pred_boxes, overlap_threshold=overlap_threshold)
    valid_pred_indices = [i for i in range(len(pred_boxes)) if i not in filtered_pred_indices]
    return [pred_boxes[i] for i in valid_pred_indices]

def load_truth_boxes(truth_path, truth_is_YOLO, width, height):
    """
    Loads and converts truth boxes if needed
    
    Parameters
    ----------
    truth_path : str
        Path to the truth file
    truth_is_YOLO : bool
        Whether the truth boxes are in YOLO format (legacy parameter can be removed since it will always be true)
    width : int
        Width of the image in pixels
    height : int
        Height of the image in pixels
    
    Returns
    -------
    truth_boxes : numpy array
        Array of truth bounding boxes in the format [x1, y1, x2, y2]
    """
    truth_boxes_raw = load_boxes(truth_path)
    
    if truth_is_YOLO:
        truth_boxes = []
        for box in truth_boxes_raw:
            x1 = int((box[0] - box[2]/2) * width)
            y1 = int((box[1] - box[3]/2) * height)
            x2 = int((box[0] + box[2]/2) * width)
            y2 = int((box[1] + box[3]/2) * height)
            truth_boxes.append([x1, y1, x2, y2])
        return np.array(truth_boxes)
    else:
        return truth_boxes_raw

def load_shapefiles_for_testing(crs, found_shp_file):
    """
    Loads found and removed building shapefiles
    
    Parameters
    ----------
    crs : pyproj.CRS
        Coordinate Reference System of the image
    found_shp_file : str
        Path to the found buildings shapefile
    
    Returns
    -------
    found_buildings : GeoDataFrame
        GeoDataFrame of found buildings
    removed_buildings : GeoDataFrame
        GeoDataFrame of removed buildings
    """
    found_buildings, removed_buildings = None, None
    
    try:
        # Load the found buildings shapefile
        found_buildings_gdf = gpd.read_file(found_shp_file)

        # Ensure CRS matches the image CRS
        if found_buildings_gdf.crs != crs:
            found_buildings_gdf = found_buildings_gdf.to_crs(crs)
        
        # Get specific types
        type_field = 'type'  # Default field name
        unique_types = found_buildings_gdf[type_field].unique()
        
        if 'new' in unique_types:
            found_buildings = found_buildings_gdf[found_buildings_gdf[type_field] == 'new']
        if 'removed' in unique_types:
            removed_buildings = found_buildings_gdf[found_buildings_gdf[type_field] == 'removed']
            
    except Exception as e:
        print(f"Error loading buildings shapefile: {e}")
    
    return found_buildings, removed_buildings

def classify_truth_boxes(truth_boxes):
    """
    Separates truth boxes into points and regular boxes based on area
    
    Parameters
    ----------
    truth_boxes : numpy array
        Array of truth bounding boxes in the format [x1, y1, x2, y2]
    
    Returns
    -------
    point_indices : list of int
        List of indices of points
    box_indices : list of int
        List of indices of regular boxes
    """
    point_indices = []
    box_indices = []
    
    for i, box in enumerate(truth_boxes):
        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height
        
        if area < 200:  # Area threshold for points
            point_indices.append(i)
        else:
            box_indices.append(i)
    
    return point_indices, box_indices

def find_box_point_pairs(truth_boxes, point_indices, box_indices):
    """
    Identifies points that are close to boxes and should be considered together. Ensures that points are not double-counted.
    This function also removes points that are associated with boxes from the point_indices list so that only the box is counted.
    
    Parameters
    ----------
    truth_boxes : numpy array
        Array of truth bounding boxes in the format [x1, y1, x2, y2]
    point_indices : list of int
        List of indices of points
    box_indices : list of int
        List of indices of regular boxes
    
    Returns
    -------
    box_point_pairs : dict
        Dictionary mapping box indices to lists of point indices
    remaining_points : list of int
        List of point indices that are not associated with any box
    """
    box_point_pairs = {}
    points_to_remove = set()
    
    for p_idx in point_indices:
        point = truth_boxes[p_idx]
        center_x = (point[0] + point[2]) / 2
        center_y = (point[1] + point[3]) / 2
        
        for b_idx in box_indices:
            box = truth_boxes[b_idx]
            
            # Check if point is near the box (slightly expanded)
            if ((box[0] - 10 <= center_x <= box[2] + 10) and 
                (box[1] - 10 <= center_y <= box[3] + 10)):
                # Associate this point with this box
                if b_idx not in box_point_pairs:
                    box_point_pairs[b_idx] = []
                box_point_pairs[b_idx].append(p_idx)
                points_to_remove.add(p_idx)
                break
    
    # Remove paired points from point_indices
    remaining_points = [p for p in point_indices if p not in points_to_remove]
    
    return box_point_pairs, remaining_points

def match_box_point_pairs(box_point_pairs, truth_boxes, pred_boxes, matched_truths, matched_predictions, annotated_image):
    """
    Process and match box-point pairs to predictions
    
    Parameters
    ----------
    box_point_pairs : dict
        Dictionary mapping box indices to lists of point indices
    truth_boxes : numpy array
        Array of truth bounding boxes in the format [x1, y1, x2, y2]
    pred_boxes : numpy array
        Array of predicted bounding boxes in the format [x1, y1, x2, y2]
    matched_truths : set
        Set of matched truth indices
    matched_predictions : set
        Set of matched prediction indices
    annotated_image : numpy array
        Annotated image for visualization
    
    Returns
    -------
    processed_pairs : set
        Set of processed point indices
    """
    processed_pairs = set()
    
    for b_idx, p_indices in box_point_pairs.items():
        box = truth_boxes[b_idx]
        max_iou = 0
        best_match = -1
        
        # Find best prediction match for this box
        for j, pred_box in enumerate(pred_boxes):
            if j not in matched_predictions:
                iou = calculate_iou(box, pred_box)
                if iou > max_iou:
                    max_iou = iou
                    best_match = j
        
        # If box is matched (using relaxed threshold), consider it a full match
        if max_iou >= 0.1:  # Lower threshold but counted as full match
            matched_truths.add(b_idx)
            matched_predictions.add(best_match)
            
            # Mark all associated points as matched too
            for p_idx in p_indices:
                matched_truths.add(p_idx)
                processed_pairs.add(p_idx)
            
            # Draw green box for matched truth
            if annotated_image is not None:
                cv2.rectangle(annotated_image, 
                            (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            (0, 255, 0), 2)
            
                # Draw the matching prediction in green too
                pred_box = pred_boxes[best_match]
                cv2.rectangle(annotated_image, 
                            (int(pred_box[0]), int(pred_box[1])), 
                            (int(pred_box[2]), int(pred_box[3])), 
                            (0, 255, 0), 1)
    
    return processed_pairs

def match_standalone_boxes(box_indices, truth_boxes, pred_boxes, matched_truths, matched_predictions, annotated_image):
    """
    Match standalone boxes (no associated points) to predictions
    
    Parameters
    ----------
    box_indices : list of int
        List of indices of box indices
    truth_boxes : numpy array
        Array of truth bounding boxes in the format [x1, y1, x2, y2]
    pred_boxes : numpy array
        Array of predicted bounding boxes in the format [x1, y1, x2, y2]
    matched_truths : set
        Set of matched truth indices
    matched_predictions : set
        Set of matched prediction indices
    annotated_image : numpy array
        Annotated image for visualization
    """
    for i in box_indices:
        if i in matched_truths:
            continue  # Skip already matched boxes
            
        box = truth_boxes[i]
        max_iou = 0
        best_match = -1
        
        # Find best prediction match
        for j, pred_box in enumerate(pred_boxes):
            if j not in matched_predictions:
                iou = calculate_iou(box, pred_box)
                if iou > max_iou:
                    max_iou = iou
                    best_match = j
        
        # If good match found (using relaxed threshold)
        if max_iou >= 0.3:
            matched_truths.add(i)
            matched_predictions.add(best_match)
            
            # Green for detected
            cv2.rectangle(annotated_image, 
                            (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            (0, 255, 0), 2)
            
            # Draw the prediction in green too
            pred_box = pred_boxes[best_match]
            cv2.rectangle(annotated_image, 
                        (int(pred_box[0]), int(pred_box[1])), 
                        (int(pred_box[2]), int(pred_box[3])), 
                        (0, 255, 0), 1)

def find_potential_point_matches(point_indices, truth_boxes, pred_boxes, processed_pairs, 
                              matched_truths, matched_predictions, point_distance_tolerance,
                              pred_confidences=None): # Add pred_confidences parameter
    """
    Find all potential matches between points and predictions
    
    Parameters
    ----------
    point_indices : list of int
        List of indices of points
    truth_boxes : numpy array
        Array of truth bounding boxes in the format [x1, y1, x2, y2]
    pred_boxes : numpy array
        Array of predicted bounding boxes in the format [x1, y1, x2, y2]
    processed_pairs : set
        Set of processed point indices
    matched_truths : set
        Set of matched truth indices
    matched_predictions : set
        Set of matched prediction indices
    point_distance_tolerance : float
        Distance tolerance for matching points to predictions
    pred_confidences : list, optional
        List of confidence scores for predictions. Required if sorting by confidence.
    
    Returns
    -------
    point_to_pred_matches : dict
        Dictionary mapping point indices to lists of potential prediction matches
    """
    point_to_pred_matches = {}  # Maps point index to [(pred_idx, distance, confidence), ...]

    for i in point_indices:
        if i in processed_pairs or i in matched_truths:
            continue  # Skip points already processed
            
        truth_point = truth_boxes[i]
        point_center_x = (truth_point[0] + truth_point[2]) / 2
        point_center_y = (truth_point[1] + truth_point[3]) / 2
        
        # Track all potential matches for this point
        potential_matches = []
        
        # Check all prediction boxes for potential matches
        for j, pred_box in enumerate(pred_boxes):
            if j in matched_predictions:
                continue
            
            current_confidence = pred_confidences[j] if pred_confidences is not None and j < len(pred_confidences) else 0.0
                
            pred_center_x = (pred_box[0] + pred_box[2]) / 2
            pred_center_y = (pred_box[1] + pred_box[3]) / 2
            
            # Check if point is inside box
            if (pred_box[0] <= point_center_x <= pred_box[2] and 
                pred_box[1] <= point_center_y <= pred_box[3]):
                # Calculate distance to center
                distance = ((point_center_x - pred_center_x)**2 + 
                            (point_center_y - pred_center_y)**2)**0.5
                potential_matches.append((j, distance, current_confidence))
                
            # Or check if point is NEAR box (using distance tolerance)
            else:
                # Calculate closest distance to box
                dx = max(pred_box[0] - point_center_x, 0, point_center_x - pred_box[2])
                dy = max(pred_box[1] - point_center_y, 0, point_center_y - pred_box[3])
                distance_to_edge = (dx**2 + dy**2)**0.5
                
                if distance_to_edge <= point_distance_tolerance:
                    # Calculate distance to center for ranking
                    distance_to_center = ((point_center_x - pred_center_x)**2 + 
                                        (point_center_y - pred_center_y)**2)**0.5
                    potential_matches.append((j, distance_to_center, current_confidence))
        
        if potential_matches:
            point_to_pred_matches[i] = potential_matches

    return point_to_pred_matches

def assign_point_matches(point_matches, truth_boxes, pred_boxes, matched_truths, matched_predictions, annotated_image):
    """Assign optimal matches between points and predictions, prioritizing confidence.
    
    Parameters
    ----------
    point_matches : dict
        Dictionary mapping point indices to lists of potential prediction matches
    truth_boxes : numpy array
        Array of truth bounding boxes in the format [x1, y1, x2, y2]
    pred_boxes : numpy array
        Array of predicted bounding boxes in the format [x1, y1, x2, y2]
    matched_truths : set
        Set of matched truth indices
    matched_predictions : set
        Set of matched prediction indices
    annotated_image : numpy array
        Annotated image for visualization
    """
    # Sort all point indices by number of potential matches (ascending)
    # This heuristic tries to match points with fewer options first.
    sorted_points = sorted(point_matches.keys(), 
                        key=lambda x: len(point_matches[x]))

    for point_idx in sorted_points:
        if point_idx in matched_truths:
            continue  # Skip if already matched in a previous iteration
            
        # Sort potential matches by confidence (descending), then by distance (ascending) as a tie-breaker.
        # Assumes point_matches[point_idx] contains tuples like (pred_idx, distance, confidence)
        matches = sorted(point_matches[point_idx], key=lambda x: (x[2], -x[1]), reverse=True) # x[2] is confidence, x[1] is distance
        
        for pred_idx, distance, confidence in matches: # Unpack confidence as well
            if pred_idx not in matched_predictions:
                # This is the best available match
                matched_truths.add(point_idx)
                matched_predictions.add(pred_idx)
                
                # Draw the match
                truth_point = truth_boxes[point_idx]
                pred_box = pred_boxes[pred_idx]
                
                # Mark in green (true positive)
                cv2.rectangle(annotated_image, 
                            (int(truth_point[0]), int(truth_point[1])), 
                            (int(truth_point[2]), int(truth_point[3])), 
                            (0, 255, 0), 2)
                
                # Also mark the matching prediction
                cv2.rectangle(annotated_image, 
                            (int(pred_box[0]), int(pred_box[1])), 
                            (int(pred_box[2]), int(pred_box[3])), 
                            (0, 255, 0), 1)
                
                # Draw a line connecting the centers to visualize the match
                point_center_x = (truth_point[0] + truth_point[2]) / 2
                point_center_y = (truth_point[1] + truth_point[3]) / 2
                pred_center_x = (pred_box[0] + pred_box[2]) / 2
                pred_center_y = (pred_box[1] + pred_box[3]) / 2
                
                cv2.line(annotated_image, 
                        (int(point_center_x), int(point_center_y)),
                        (int(pred_center_x), int(pred_center_y)),
                        (0, 255, 255), 1)  # Yellow line
                
                break  # Stop after finding the best match

def process_removed_buildings(missed, removed_buildings, transform, inverse_transform, 
                           bounds_geo, pred_boxes, matched_predictions, annotated_image):
    """
    Process removed buildings shapefile and mark missed detections
    
    Parameters
    ----------
    missed : list of lists
        List of missed building boxes in pixel coordinates [x1, y1, x2, y2]
    removed_buildings : GeoDataFrame
        GeoDataFrame of removed buildings
    transform : pyproj.Transformer
        Transformer for converting pixel coordinates to geographic coordinates
    inverse_transform : pyproj.Transformer
        Inverse transformer for converting geographic coordinates to pixel coordinates
    bounds_geo : tuple
        Geographic bounds of the image (left, bottom, right, top)
    pred_boxes : list of lists
        List of predicted building boxes in pixel coordinates [x1, y1, x2, y2]
    matched_predictions : set
        Set of matched prediction indices
    annotated_image : numpy array
        Annotated image for visualization
    
    Returns
    -------
    matching_indices : list of int
        List of indices of missed boxes that match removed buildings
    """
    # Create building geometries from missed buildings only
    missed_geoms = []
    for box in missed:  
        x1, y1, x2, y2 = map(int, box)
        
        # Transform to geographic coordinates
        ul_x, ul_y = transform * (x1, y1)  # Upper left
        lr_x, lr_y = transform * (x2, y2)  # Lower right
        
        # Create polygon for building footprint
        missed_poly = shp_box(ul_x, ul_y, lr_x, lr_y)
        missed_geoms.append(missed_poly)
    
    # Check each building geometry for intersection with removed buildings
    removed_matches = []  # Stores (missed_index, removed_geom_index, intersection_area)
    
    # Track which removed buildings have matches
    matched_removed_buildings = set()
    
    # Find intersections between missed boxes and removed buildings
    for i, geom in enumerate(missed_geoms):
        for removed_idx, removed_geom in enumerate(removed_buildings.geometry):
            if geom.is_valid and removed_geom.is_valid and geom.intersects(removed_geom):
                try:
                    intersection = geom.intersection(removed_geom)
                    if not intersection.is_empty:
                        intersection_area = intersection.area
                        if intersection_area > 1e-9:  # Use threshold to avoid floating point issues
                            removed_matches.append((i, removed_idx, intersection_area))
                            matched_removed_buildings.add(removed_idx)
                except Exception:
                    pass  # Handle invalid geometries silently
    
    # Find best matches based on intersection area
    best_match_for_removed = {}  # Maps removed_idx -> (missed_index, area)
    for miss_idx, removed_idx, area in removed_matches:
        if removed_idx not in best_match_for_removed or area > best_match_for_removed[removed_idx][1]:
            best_match_for_removed[removed_idx] = (miss_idx, area)
    
    # Get unique missed box indices that are the best match
    missed_indices_to_mark = {match[0] for match in best_match_for_removed.values()}
    
    # Mark these best-matching missed boxes
    matching_indices = []  # Track which original truth indices match
    
    for missed_idx in missed_indices_to_mark:
        if 0 <= missed_idx < len(missed):
            # Get the corresponding box from missed
            box = missed[missed_idx]
            
            # Draw blue rectangle for removed building match
            cv2.rectangle(annotated_image, 
                    (int(box[0]), int(box[1])), 
                    (int(box[2]), int(box[3])), 
                    (130, 0, 78), 3)  # Blue for removed buildings

            # Add to tracking list
            matching_indices.append(missed_idx)
    
    # Draw removed buildings from shapefile
    image_polygon = shp_box(*bounds_geo)
    visible_unmatched_count = 0
    
    for removed_idx in range(len(removed_buildings)):
        removed_geom = removed_buildings.iloc[removed_idx].geometry
        
        # Check if this geometry intersects the image
        if removed_geom.intersects(image_polygon):
            # Get the intersection with the image
            visible_geom = removed_geom.intersection(image_polygon)
            
            # Check if this removed building intersects with any prediction
            intersects_prediction = check_prediction_intersection(
                visible_geom, pred_boxes, transform, matched_predictions
            )
            
            # Draw the geometry based on its type and match status
            draw_geometry(
                visible_geom, inverse_transform, annotated_image,
                removed_idx in matched_removed_buildings, intersects_prediction,
                color_matched=(255, 255, 0),  # Cyan
                color_intersects=(0, 255, 255)  # Yellow
            )
            
            if intersects_prediction:
                visible_unmatched_count += 1
    
    # Add legend for removed buildings with false predictions
    if visible_unmatched_count > 0:
        cv2.putText(annotated_image, f"Removed building with false prediction", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Yellow
    
    return matching_indices

def check_prediction_intersection(geometry, pred_boxes, transform, matched_predictions):
    """
    Check if a geometry intersects with any unmatched prediction
    
    Parameters
    ----------
    geometry : shapely.geometry
        Geometry object to check for intersection
    pred_boxes : list of lists
        List of predicted bounding boxes in the format [x1, y1, x2, y2]
    transform : pyproj.Transformer
        Transformer for converting pixel coordinates to geographic coordinates
    matched_predictions : set
        Set of matched prediction indices
    """
    for pred_idx, pred_box in enumerate(pred_boxes):
        if pred_idx in matched_predictions:
            continue  # Skip already matched predictions
        
        # Convert prediction box to geometry
        x1, y1, x2, y2 = map(int, pred_box)
        pred_ul_x, pred_ul_y = transform * (x1, y1)
        pred_lr_x, pred_lr_y = transform * (x2, y2)
        pred_poly = shp_box(pred_ul_x, pred_ul_y, pred_lr_x, pred_lr_y)
        
        # Check for intersection
        if pred_poly.is_valid and geometry.is_valid and pred_poly.intersects(geometry):
            try:
                intersection = pred_poly.intersection(geometry)
                if not intersection.is_empty and intersection.area > 1e-9:
                    return True
            except Exception:
                pass
    
    return False

def draw_geometry(geometry, inverse_transform, image, is_matched, intersects_prediction, 
                color_matched, color_intersects):
    """Draw a geometry on the image with appropriate color"""
    if geometry.geom_type == 'Polygon':
        pixel_coords = []
        for x, y in geometry.exterior.coords:
            px, py = inverse_transform * (x, y)
            pixel_coords.append((int(px), int(py)))
        
        if len(pixel_coords) > 2:  # Need at least 3 points for a polygon
            if is_matched:
                cv2.polylines(image, [np.array(pixel_coords)], 
                            True, color_matched, 3)
            elif intersects_prediction:
                cv2.polylines(image, [np.array(pixel_coords)], 
                            True, color_intersects, 3)
    
    elif geometry.geom_type == 'MultiPolygon':
        for poly in geometry.geoms:
            poly_pixels = []
            for x, y in poly.exterior.coords:
                px, py = inverse_transform * (x, y)
                poly_pixels.append((int(px), int(py)))
            
            if len(poly_pixels) > 2:
                if is_matched:
                    cv2.polylines(image, [np.array(poly_pixels)], 
                                True, color_matched, 3)
                elif intersects_prediction:
                    cv2.polylines(image, [np.array(poly_pixels)], 
                                True, color_intersects, 3)

def get_filtered_false_positives(pred_boxes, matched_predictions, pred_confidences, fp_confidence):
    """Get false positives and filter by confidence threshold"""
    false_positive = []
    fp_confidences = []
    
    for i, (box, conf) in enumerate(zip(pred_boxes, pred_confidences)):
        if i not in matched_predictions:
            false_positive.append(box)
            fp_confidences.append(conf)
    
    # Filter false positives based on confidence threshold
    confidence_threshold = fp_confidence
    filtered_fp = []
    filtered_fp_conf = []

    for i, (box, conf) in enumerate(zip(false_positive, fp_confidences)):
        if conf >= confidence_threshold:
            filtered_fp.append(box)
            filtered_fp_conf.append(conf)

    return filtered_fp, filtered_fp_conf

def process_found_buildings(filtered_fp, found_buildings, transform, inverse_transform,
                          bounds_geo, pred_boxes, annotated_image):
    """Process found buildings shapefile and mark matching false positives"""
    # Create building geometries from false positives
    building_geoms = []
    for box in filtered_fp:  
        x1, y1, x2, y2 = map(int, box)
        
        # Transform to geographic coordinates
        ul_x, ul_y = transform * (x1, y1)
        lr_x, lr_y = transform * (x2, y2)
        
        # Create polygon for building footprint
        building_poly = shp_box(ul_x, ul_y, lr_x, lr_y)
        building_geoms.append(building_poly)
    
    # Find intersections between false positives and found buildings
    found_matches = []
    matched_found_buildings = set()
    
    for i, geom in enumerate(building_geoms):
        for found_idx, found_geom in enumerate(found_buildings.geometry):
            if geom.is_valid and found_geom.is_valid and geom.intersects(found_geom):
                try:
                    intersection = geom.intersection(found_geom)
                    if not intersection.is_empty:
                        intersection_area = intersection.area
                        if intersection_area > 1e-9:
                            found_matches.append((i, found_idx, intersection_area))
                            matched_found_buildings.add(found_idx)
                except Exception:
                    pass
    
    # Find best matches based on intersection area
    best_match_for_found = {}
    for fp_idx, found_idx, area in found_matches:
        if found_idx not in best_match_for_found or area > best_match_for_found[found_idx][1]:
            best_match_for_found[found_idx] = (fp_idx, area)
    
    # Get unique false positive indices that are best matches
    fp_indices_to_mark = {match[0] for match in best_match_for_found.values()}
    
    # Mark these best-matching false positives in blue
    matching_indices = []
    
    for fp_idx in fp_indices_to_mark:
        if 0 <= fp_idx < len(filtered_fp):
            box = filtered_fp[fp_idx]
            
            # Find original index in pred_boxes
            orig_indices = [j for j, pb in enumerate(pred_boxes) if np.array_equal(pb, box)]
            if orig_indices:
                orig_idx = orig_indices[0]
                matching_indices.append(orig_idx)
                
                # Draw blue rectangle for found building match
                cv2.rectangle(annotated_image, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (255, 0, 0), 3)  # Blue for found buildings
    
    # Draw unmatched found buildings from shapefile in orange
    unmatched_found_buildings = [i for i in range(len(found_buildings)) if i not in matched_found_buildings]
    
    if unmatched_found_buildings:
        image_polygon = shp_box(*bounds_geo)
        visible_unmatched_count = 0
        
        for found_idx in unmatched_found_buildings:
            found_geom = found_buildings.iloc[found_idx].geometry
            
            if found_geom.intersects(image_polygon):
                visible_geom = found_geom.intersection(image_polygon)
                
                # Draw the geometry in orange
                draw_unmatched_geometry(visible_geom, inverse_transform, annotated_image, (0, 165, 255))
                visible_unmatched_count += 1
        
        if visible_unmatched_count > 0:
            cv2.putText(annotated_image, f"Building in shapefile (no match)", (20, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)  # Orange
    
    return matching_indices

def draw_unmatched_geometry(geometry, inverse_transform, image, color):
    """Draw an unmatched geometry on the image"""
    if geometry.geom_type == 'Polygon':
        pixel_coords = []
        for x, y in geometry.exterior.coords:
            px, py = inverse_transform * (x, y)
            pixel_coords.append((int(px), int(py)))
        
        if len(pixel_coords) > 2:
            cv2.polylines(image, [np.array(pixel_coords)], True, color, 3)
    
    elif geometry.geom_type == 'MultiPolygon':
        for poly in geometry.geoms:
            poly_pixels = []
            for x, y in poly.exterior.coords:
                px, py = inverse_transform * (x, y)
                poly_pixels.append((int(px), int(py)))
            
            if len(poly_pixels) > 2:
                cv2.polylines(image, [np.array(poly_pixels)], True, color, 3)

def add_found_buildings_legend(image, output_dir, found_box_matches):
    """Add legend for found buildings and write to output file"""
    cv2.putText(image, f"Found in shapefile", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Write to output file
    found_buildings_file = Path(output_dir) / "found_buildings_images.txt"
    with open(found_buildings_file, 'a') as f:
        f.write(f"{found_box_matches['image_id']}: {len(found_box_matches['matching_boxes'])} matches\n")

def add_removed_buildings_legend(image, output_dir, removed_box_matches):
    """Add legend for removed buildings and write to output file"""
    cv2.putText(image, f"Removed in shapefile", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (130, 0, 78), 2)  # Blue
    
    # Add legend for shapefile buildings
    cv2.putText(image, f"Removed building footprints", (20, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # Cyan
    
    # Write to output file
    removed_buildings_file = Path(output_dir) / "removed_buildings_images.txt"
    with open(removed_buildings_file, 'a') as f:
        f.write(f"{removed_box_matches['image_id']}: {len(removed_box_matches['matching_boxes'])} matches\n")

def mark_false_positives(false_positive, fp_confidences, pred_boxes, image, matched_found_boxes):
    """Mark false positives in purple with confidence scores"""
    for i, (box, conf) in enumerate(zip(false_positive, fp_confidences)):
        # Convert box index in filtered list back to original index
        orig_idx = [j for j, pb in enumerate(pred_boxes) if np.array_equal(pb, box)][0]
        
        # Skip purple coloring if this box is matched in found buildings
        if orig_idx not in matched_found_boxes:
            # Mark in purple (false positive)
            cv2.rectangle(image, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (255, 0, 255), 2)
            
            # Add confidence text
            conf_text = f"{conf:.2f}"
            cv2.putText(image, conf_text,
                      (int(box[0]), int(box[1])-5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

def visualize_pipeline_and_buffer(image, pipeline_shp_path, crs, transform, inverse_transform, 
                                bounds_geo, height, width, max_distance):
    """Visualize pipeline and buffer zone on the image"""
    try:
        # Load pipeline shapefile
        pipeline_gdf = gpd.read_file(pipeline_shp_path)
        
        # Ensure CRS matches
        if pipeline_gdf.crs != crs:
            pipeline_gdf = pipeline_gdf.to_crs(crs)
        
        # Get pipeline geometry
        pipeline_geom = pipeline_gdf.geometry.union_all('unary')
        
        # Get image bounds
        image_polygon = shp_box(*bounds_geo)
        
        # Create buffer for visualization
        buffer_geom = create_pipeline_buffer(pipeline_geom, crs, max_distance)
        
        # Visualize buffer (corridor)
        if buffer_geom.intersects(image_polygon):
            buffer_in_image = buffer_geom.intersection(image_polygon)
            
            # Create buffer mask
            buffer_mask = create_buffer_mask(buffer_in_image, transform, height, width)
            
            # Apply colored overlay for buffer
            buffer_pixels = np.count_nonzero(buffer_mask)
            if buffer_pixels > 0:
                buffer_color = np.zeros_like(image)
                buffer_color[buffer_mask > 0] = [172, 0, 0]  # Light blue
                
                # Apply with transparency
                alpha = 0.3
                image = cv2.addWeighted(image, 1.0, buffer_color, alpha, 0)
                
        # Check if pipeline intersects with image
        if pipeline_geom.intersects(image_polygon):
            # Draw pipeline on top
            pipeline_in_image = pipeline_geom.intersection(image_polygon)
            draw_pipeline(pipeline_in_image, inverse_transform, image)
            
            # Add legend
            cv2.putText(image, f"Pipeline", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f"Buffer zone ({max_distance}m)", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 144, 255), 2)
    
    except Exception as e:
        print(f"Error visualizing pipeline: {e}")
    
    return image

def create_pipeline_buffer(pipeline_geom, crs, max_distance):
    """Create buffer around pipeline with appropriate CRS handling"""
    if crs.is_projected:
        return pipeline_geom.buffer(max_distance)
    else:
        # Create geodesic buffer for geographic CRS
        try:
            # Get UTM zone for pipeline center
            pipeline_centroid = pipeline_geom.centroid
            utm_band = int((pipeline_centroid.x + 180) / 6) + 1
            utm_epsg = 32600 + utm_band if pipeline_centroid.y >= 0 else 32700 + utm_band
            utm_crs = pyproj.CRS.from_epsg(utm_epsg)
            
            # Transform to UTM, buffer, and back
            project_to_utm = pyproj.Transformer.from_crs(crs, utm_crs, always_xy=True).transform
            project_to_orig = pyproj.Transformer.from_crs(utm_crs, crs, always_xy=True).transform
            
            pipeline_utm = shapely_transform(project_to_utm, pipeline_geom)
            buffer_utm = pipeline_utm.buffer(max_distance)
            return shapely_transform(project_to_orig, buffer_utm)
        except Exception as e:
            print(f"Couldn't create geodesic buffer: {e}")
            # Fallback to approximate degrees
            degree_distance = max_distance / 111000
            return pipeline_geom.buffer(degree_distance)

def create_buffer_mask(buffer_geom, transform, height, width):
    """Create a mask for the buffer area"""
    buffer_mask = np.zeros((height, width), dtype=np.uint8)
    step = 5  # Check every 5 pixels for performance
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            geo_x, geo_y = transform * (x, y)
            point = Point(geo_x, geo_y)
            if buffer_geom.contains(point):
                cv2.rectangle(buffer_mask, (x-step//2, y-step//2), 
                              (x+step//2, y+step//2), 255, -1)
    
    return buffer_mask

def draw_pipeline(pipeline_geom, inverse_transform, image):
    """Draw pipeline on the image"""
    if pipeline_geom.geom_type == 'LineString':
        pixel_coords = []
        for x, y in pipeline_geom.coords:
            px, py = inverse_transform * (x, y)
            pixel_coords.append((int(px), int(py)))
        
        # Draw pipeline as thick yellow line
        for i in range(len(pixel_coords) - 1):
            cv2.line(image, pixel_coords[i], pixel_coords[i+1], 
                     (0, 255, 255), 3)
    
    elif pipeline_geom.geom_type == 'MultiLineString':
        for line in pipeline_geom.geoms:
            pixel_coords = []
            for x, y in line.coords:
                px, py = inverse_transform * (x, y)
                pixel_coords.append((int(px), int(py)))
            
            for i in range(len(pixel_coords) - 1):
                cv2.line(image, pixel_coords[i], pixel_coords[i+1], 
                         (0, 255, 255), 3)

# Function to calculate proper distance in meters
def calculate_distance(point, line_geom, crs):
    if crs.is_projected:
        # For projected CRS, use regular Shapely distance (should be in meters or the CRS's unit)
        return point.distance(line_geom)
    else:
        # For geographic CRS (like WGS84), use geodesic distance
        geod = pyproj.Geod(ellps='WGS84')
        
        # Find the closest point on the line to our point
        p1, p2 = nearest_points(point, line_geom)
        
        # Calculate geodesic distance
        _, _, distance = geod.inv(p1.x, p1.y, p2.x, p2.y)
        return distance

def process_images_with_saved_predictions(image_files, predictions, confidences, output_dir, use_pipeline, pipeline_path, found_shp_file,
                                          max_distance, fp_confidence, model_version, model_type, testing=False, save_images=True):
    """Process images using saved predictions"""
    # Create output directory with timestamp so you don't overwrite previous results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir) / f"analysis_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    areas_missed = []
    areas_partial = []
    all_missed_metrics = []
    
    total_missed = 0
    total_partial = 0
    total_fp = 0
    total_buildings = 0
    
    # New metric counters
    total_shapefile_objects_considered_found = 0
    shapefile_objects_auto_found_in_flagged_images = 0
    shapefile_objects_matched_in_non_flagged_images = 0

    # Track images with buildings matching the found shapefile
    images_with_found_buildings = []
    images_with_missed_buildings = []
    
    # Track flagged images and their metrics
    flagged_images = []
    flagged_missed = 0
    flagged_fp = 0
    flagged_buildings = 0
    flagged_shapefile_matches_count = 0
    
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Skip if no predictions found
        if str(img_path) not in predictions:
            print(f"No predictions found for {img_path.name}")
            continue
            
        # Get predictions
        pred_boxes = predictions[str(img_path)]
        pred_confidences = confidences[str(img_path)] if str(img_path) in confidences else None
        
        # Get truth path
        truth_path = Path(img_path.parent.parent) / 'labels' / f"{img_path.stem}.txt"
        if not truth_path.exists():
            print(f"Warning: No truth file found: {truth_path}")
            continue
            
        # Count total buildings
        with open(truth_path) as f:
            buildings = len(f.readlines())
            total_buildings += buildings
        
        # Save predictions temporarily for custom_analyze_missed_boxes function
        pred_path = results_dir / f"{img_path.stem}_pred.txt"
        with open(pred_path, 'w') as f:
            for box in pred_boxes:
                box_coords = ' '.join(map(str, box.tolist()))
                f.write(f"{box_coords}\n")
        
        # Call custom analysis
        pipeline_path = pipeline_path if use_pipeline else None
        missed, false_positive, analysis_image_path, _, found_matches, reported_changes, \
        shapefile_objects_actually_matched, count_missed_reported = post_processing_analysis(
            pred_path, 
            truth_path,
            img_path,
            results_dir,
            truth_is_YOLO=True,
            pipeline_shp_path=pipeline_path,
            found_shp_file=found_shp_file,
            max_distance=max_distance,
            pred_confidences=pred_confidences,
            fp_confidence=fp_confidence,
            testing=testing,
            save_images=save_images
        )
        
        # Collect area information and update counts (regardless of save_images)
        areas_missed.extend((box[2] - box[0]) * (box[3] - box[1]) for box in missed)
        
        
        # Calculate metrics if images are saved
        if save_images and analysis_image_path:
            missed_metrics = calculate_box_metrics(analysis_image_path, missed)
            all_missed_metrics.extend([{**m, 'image': img_path.name} for m in missed_metrics])
        
        # Track counts
        total_missed += len(missed)
        
        total_fp += len(false_positive)
        is_flagged = False    
        # Flag images with high error rates
        if (buildings > 10) or (buildings < 10 and buildings + len(false_positive) > 10):
            
            if (buildings > 0 and (len(false_positive)/buildings > 0.3 or len(missed)/buildings > 0.3)) or (buildings == 0 and (len(false_positive) > 0 or len(missed) > 0)): # in case buildings == 0
                is_flagged = True
                # print(f"Warning: High false positive/missed rate for {img_path.name}: {len(false_positive)/buildings:.2f} / {len(missed)/buildings:.2f}")
                # Write the flagged image name to flags.txt
                flags_file_path = results_dir / "flags.txt"
                with open(flags_file_path, 'a') as f_flags:
                    f_flags.write(f"{img_path.name}\n")
                
                # Track metrics for flagged images
                flagged_images.append({
                    "image_id": img_path.name,
                    "buildings": buildings,
                    "missed": len(missed),
                    "false_positive": len(false_positive),
                    "fp_rate": len(false_positive)/buildings if buildings > 0 else 0,
                    "missed_rate": len(missed)/buildings if buildings > 0 else 0,
                })
                
                # Update flagged totals
                flagged_missed += len(missed)
                flagged_fp += len(false_positive)
                flagged_buildings += buildings

                # For flagged images, all shapefile objects within bounds are considered "found"
                num_sf_objects_this_image = 0
                if reported_changes is not None and not reported_changes.empty:
                    num_sf_objects_this_image = len(reported_changes)
                    print(f" - Flagged image {img_path.name}: All {num_sf_objects_this_image} shapefile objects within bounds are considered found.")
                
                total_shapefile_objects_considered_found += num_sf_objects_this_image
                shapefile_objects_auto_found_in_flagged_images += num_sf_objects_this_image
                
                if num_sf_objects_this_image > 0:
                    images_with_found_buildings.append({
                        "image_id": img_path.name,
                        "auto_found_shapefile_objects": num_sf_objects_this_image
                    })
            
            else: # Not flagged
                total_shapefile_objects_considered_found += shapefile_objects_actually_matched
                shapefile_objects_matched_in_non_flagged_images += shapefile_objects_actually_matched

        # Track images with found building matches (regardless of save_images)
        if found_matches and found_matches.get("matching_boxes"):
            images_with_found_buildings.append({
                "image_id": found_matches["image_id"],
                "num_matches": len(found_matches["matching_boxes"])
            })
        if testing and count_missed_reported > 0:
            images_with_missed_buildings.append({
                "image_id": img_path.name,
                "num_matches": count_missed_reported
            })

    # Calculate performance metrics
    tp = total_buildings - total_missed
    fn = total_missed
    fp = total_fp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # tp + fn is total_buildings
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate flagged image metrics
    flagged_percentage = (len(flagged_images) / len(image_files)) * 100 if image_files else 0
    flagged_building_percentage = (flagged_buildings / total_buildings) * 100 if total_buildings > 0 else 0
    flagged_missed_percentage = (flagged_missed / total_missed) * 100 if total_missed > 0 else 0
    flagged_fp_percentage = (flagged_fp / total_fp) * 100 if total_fp > 0 else 0

    # Always create the metrics dictionary
   
    metrics_dict = {
        "True Positives (TP)": tp,
        "False Negatives (FN)": fn,
        "False Positives (FP)": fp,
        "Total Ground Truth": total_buildings,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Flagged Images Count": len(flagged_images),
        "Flagged Images Percentage": flagged_percentage,
        "Flagged Buildings Count": flagged_buildings,
        "Flagged Buildings Percentage": flagged_building_percentage,
        "Flagged Missed Count": flagged_missed,
        "Flagged Missed Percentage": flagged_missed_percentage,
        "Flagged FP Count": flagged_fp,
        "Flagged FP Percentage": flagged_fp_percentage,
        "Shapefile Objects Auto-Found in Flagged Images": shapefile_objects_auto_found_in_flagged_images,
        "Shapefile Objects Matched by Model in Non-Flagged Images": shapefile_objects_matched_in_non_flagged_images
        
    }

    if save_images:
        # Create summary visualizations
        create_final_histogram(areas_missed, results_dir)
        create_final_pie_chart(total_missed, total_buildings, results_dir)
    
        # Save metrics
        if all_missed_metrics:
            pd.DataFrame(all_missed_metrics).to_csv(
                f'{results_dir}/box_metrics_{model_type}{"-"+ model_version if model_version else ""}.csv', 
                index=False
            )
            
        # Create performance summary DataFrame
        performance_summary = pd.DataFrame({
            'Metric': list(metrics_dict.keys()),
            'Value': list(metrics_dict.values())
        })
        
        # Define performance summary filename
        model_suffix = f"-{model_version}" if model_version else ""
        performance_filename = f'performance_summary_{model_type}{model_suffix}.csv'
        performance_filepath = results_dir / performance_filename
        
        # Save performance summary
        performance_summary.to_csv(performance_filepath, index=False)
        print(f"Performance summary saved to: {performance_filepath}")
        
        # Save detailed flagged images data if any exist
        if flagged_images:
            flagged_df = pd.DataFrame(flagged_images)
            flagged_filepath = results_dir / f'flagged_images_{model_type}{model_suffix}.csv'
            flagged_df.to_csv(flagged_filepath, index=False)
            print(f"Flagged images details saved to: {flagged_filepath}")
        
        # Print results summary
        print(f"\nAnalysis complete:\n")
        print(f"Results saved to: {results_dir}")
        print(f"Total buildings: {total_buildings}")
        print(f"Correct detections: {tp} ({tp/total_buildings*100:.1f}%)")
        print(f"  - Perfect matches: {tp - total_partial}")
        print(f"  - Partial matches: {total_partial} (counted as correct)")
        print(f"Missed detections: {fn} ({fn/total_buildings*100:.1f}%)")
        print(f"New Detections: {fp}")
        
        # Print flagged images summary
        if flagged_images:
            print(f"\nFlagged Images Summary:")
            print(f"  - {len(flagged_images)} images flagged ({flagged_percentage:.1f}% of all images)")
            print(f"  - {flagged_buildings} buildings in flagged images ({flagged_building_percentage:.1f}% of all buildings)")
            print(f"  - {flagged_missed} missed detections in flagged images ({flagged_missed_percentage:.1f}% of all missed)")
            print(f"  - {flagged_fp} false positives in flagged images ({flagged_fp_percentage:.1f}% of all FP)")
        
        # Print summary of found buildings if testing was enabled
        if testing and images_with_found_buildings:
            # Sort by number of matches (descending)
            # images_with_found_buildings.sort(key=lambda x: x["num_matches"], reverse=True)
            
            print("\nFound buildings analysis:")
            print(f"  - {len(images_with_found_buildings)} images have buildings in the shapefile")
            
            # Save the list of images with found buildings to CSV
            found_buildings_df = pd.DataFrame(images_with_found_buildings)
            found_buildings_filepath = results_dir / "images_with_found_buildings.csv"
            found_buildings_df.to_csv(found_buildings_filepath, index=False)
            print(f"  - Full list saved to: {found_buildings_filepath}")

        if testing and images_with_missed_buildings:
            # images_with_missed_buildings.sort(key=lambda x: x["num_matches"], reverse=True)

            print("\n Missed buildings analysis:")
            print(f"  - {len(images_with_missed_buildings)} images have missed buildings in the shapefile")
            # Save the list of images with missed buildings to CSV
            missed_buildings_df = pd.DataFrame(images_with_missed_buildings)
            missed_buildings_filepath = results_dir / "images_with_missed_buildings.csv"
            missed_buildings_df.to_csv(missed_buildings_filepath, index=False)
            print(f"  - Full list saved to: {missed_buildings_filepath}")
    else:
        # Just print a simple summary when not saving images
        print("\nAnalysis complete (metrics only):")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1_score:.3f}")
        if flagged_images:
            print(f"Flagged: {len(flagged_images)} images with {flagged_missed} missed and {flagged_fp} FP")
    
    # Always return both the results directory and metrics
    return results_dir, metrics_dict

# --- SLIDING WINDOW FUNCTIONS ---

def sliding_window_detection(
    models,
    images: list,  # Changed to a list of images
    window_size: int,
    overlap_ratio: float,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    output_dir: str = None,
    debug = False
) -> tuple:
    """
    Perform object detection using a sliding window approach with an ensemble of models.
    
    Parameters
    -----------
    models : list
        List of detection models (ensemble from k-fold cross validation)
    images : list
        List of images to process (numpy arrays)
    window_size : int
        Size of the square sliding window
    overlap_ratio : float
        Amount of overlap between adjacent windows (0.0 to 1.0)
    conf_threshold : float
        Confidence threshold for detections
    iou_threshold : float
        IoU threshold for NMS
    debug : bool
        Whether to save debug images
        
    Returns
    --------
    all_final_boxes : list of lists
        List of detection boxes for each image
    all_final_confidences : list of lists
        List of confidence scores for each box in each image
    """
    # If a single image is passed, convert to a list
    if not isinstance(images, list):
        images = [images]
    
    all_final_boxes = {}
    all_final_confidences = {}
    
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    prediction_dir = output_dir / f"sw_predictions_{window_size}_{overlap_ratio}"
    prediction_dir.mkdir(exist_ok=True, parents=True)
    
    # Create debug directory if debug is enabled
    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each image
    img_idx = 0
    for img_path in tqdm(images, desc="Generating predictions using the sliding window approach"):
        pred_path = prediction_dir / f"{img_path.stem}_pred.npy"
        conf_path = prediction_dir / f"{img_path.stem}_conf.npy"
        # Get image dimensions
        image = cv2.imread(str(img_path))
        height, width = image.shape[:2]
        
        # Calculate step size based on window_size and overlap_ratio
        step_size = int(window_size * (1 - overlap_ratio))
        
        # Lists to store all detections for this image
        all_boxes = []
        all_scores = []
        
        window_count = 0
        
        # Slide window across the image
        y_border = False
        for y in range(0, height, step_size):
            x_border = False
            for x in range(0, width, step_size):
                # Extract the window
                
                
                if x + window_size > width:
                    x = width - window_size
                    x_border = True
                if y + window_size > height:
                    y = height - window_size
                    y_border = True
                window = image[y:y+window_size, x:x+window_size]
                # Save debug window if requested
                if debug:
                    window_path = debug_dir / f"img{img_idx}_window{window_count}.jpg"
                    cv2.imwrite(str(window_path), window)
                
                # Process each model in the ensemble
                pred_boxes = []
                pred_confidences = []
                for model in models:
                    # Run detection on the window
                    results = model.predict(window, conf=conf_threshold, imgsz=window_size,
                                           verbose=False)
                    if x_border:
                        print(results)
                    for result in results:
                        # Extract boxes and confidences
                        pred_boxes.extend(result.boxes.xyxy.cpu().numpy())
                        pred_confidences.extend(result.boxes.conf.cpu().numpy())
                
                # Apply NMS to window predictions
                boxes, confidences = non_max_suppression(pred_boxes, pred_confidences, iou_threshold)  
                        
                idx = 0    
                # Process each detection from this model
                for box in boxes:
                    x1, y1, x2, y2 = box[:4]
                    # Adjust coordinates to the original image
                    
                    x1 += x
                    y1 += y
                    x2 += x
                    y2 += y
                            
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                            
                    if x2 > x1 and y2 > y1:  # Ensure valid box dimensions
                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(confidences[idx])
                    idx += 1
                
                window_count += 1
                if x_border:
                    break
            if y_border:
                break
        
        # Apply NMS if there are any detections
        final_boxes, final_confidences = non_max_suppression(all_boxes, all_scores, iou_threshold)
        
        all_final_boxes[str(img_path)] = final_boxes
        all_final_confidences[str(img_path)] = final_confidences
        np.save(pred_path, final_boxes)
        np.save(conf_path, final_confidences)
        img_idx += 1
    
    return all_final_boxes, all_final_confidences

def generate_sw_predictions(image_files, output_dir, models, conf_threshold, window_size=1024, overlap_ratio=0.5):
    """Load previously saved predictions"""
    sw_predictions = {}
    sw_confidences = {}
    prediction_dir = Path(output_dir) / f"sw_predictions_{window_size}_{overlap_ratio}"
    if not prediction_dir.exists():
        print(f"Warning: Prediction directory does not exist: {prediction_dir}")
        sw_predictions, sw_confidences = sliding_window_detection(
            models,
            image_files,
            window_size=1024,
            overlap_ratio=0.5,
            conf_threshold=conf_threshold,
            iou_threshold=0.2,
            output_dir=output_dir,
            debug=False
        )
    else:
        print(f"Loading predictions from: {prediction_dir}")
        for img_path in image_files:
            pred_path = prediction_dir / f"{img_path.stem}_pred.npy"
            conf_path = prediction_dir / f"{img_path.stem}_conf.npy"
            if pred_path.exists():
                pred_boxes = np.load(pred_path)
                sw_predictions[str(img_path)] = pred_boxes
            else:
                print(f"Warning: No saved predictions for {img_path.stem}")
            if conf_path.exists():
                pred_confidences = np.load(conf_path)
                sw_confidences[str(img_path)] = pred_confidences
            else:
                print(f"Warning: No saved confidences for {img_path.stem}")

        print(f"Loaded predictions for {len(sw_predictions)} images")
    return sw_predictions, sw_confidences


# --- OTHER FUNCTIONS ---
def filter_athletic_fields(shapefile_path):
    """
    Filter out athletic fields from a shapefile based on the 'Type' field.
    
    Parameters:
    -----------
    shapefile_path : str
        Path to the shapefile
        
    Returns:
    --------
    gdf : GeoDataFrame
        Filtered GeoDataFrame without athletic fields
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Check if the 'Type' field exists
    if 'Type' in gdf.columns:
        # Filter out athletic fields (case insensitive)
        mask = ~gdf['Type'].str.lower().str.contains('athletic field', na=False)
        filtered_gdf = gdf[mask]
        print(f"Filtered out {len(gdf) - len(filtered_gdf)} athletic fields from {len(gdf)} total features")
        return filtered_gdf
    else:
        print("Warning: 'Type' field not found in shapefile")
        return gdf
    
def count_type(shapefile_path, query):
    """
    Count all instances of a type in a shapefile based on the 'type' field without filtering.
    
    Parameters:
    -----------
    shapefile_path : str
        Path to the shapefile
    
    query : str
        Type to count (case sensitive)
        
    Returns:
    --------
    int
        Number of matches found
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Check if the 'Type' field exists
    if 'type' in gdf.columns:
        # Count athletic fields (case sensitive)
        mask = gdf['type'].str.contains(query, na=False)
        count = mask.sum()
        print(f"Found {count} {query} out of {len(gdf)} total features")
        return count
    else:
        print("Warning: 'Type' field not found in shapefile")
        return 0
    
def analyze_single_image(image_files, output_dir, pipeline_path, predictions, confidences, img_idx=0, use_pipeline=True, max_distance=100, fp_confidence = 0.5): # taken directly from original code so might need updating 
    """Analyze a single image in detail"""
    if img_idx >= len(image_files):
            print(f"Image index {img_idx} is out of range (max: {len(image_files)-1})")
            return
            
    img_path = image_files[img_idx]
    print(f"Analyzing image: {img_path}")
        
    # Get predictions

    if str(img_path) not in predictions:
        print(f"No predictions found for {img_path}")
        return
    pred_boxes = predictions[str(img_path)]
    pred_confidences = confidences[str(img_path)] if str(img_path) in confidences else None

    # Create a temporary output directory for this analysis
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    single_img_dir = Path(output_dir) / f"single_image_{timestamp}"
    single_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions temporarily
    pred_path = single_img_dir / f"{img_path.stem}_pred.txt"
    with open(pred_path, 'w') as f:
        for box in pred_boxes:
            box_coords = ' '.join(map(str, box.tolist()))
            f.write(f"{box_coords}\n")
            
    # Get truth path
    truth_path = Path(img_path.parent.parent) / 'labels' / f"{img_path.stem}.txt"
    
    # Run analysis
    pipeline_path = pipeline_path if use_pipeline else None
    missed, false_positive, analysis_image_path, fp_confidences,_,_,_,_ = post_processing_analysis(
        pred_path, 
        truth_path,
        img_path,
        single_img_dir,
        truth_is_YOLO=True,
        pipeline_shp_path=pipeline_path,
        max_distance=max_distance,
        pred_confidences=pred_confidences,
        testing=False,
        fp_confidence=fp_confidence  
    )
    
    
    # Print statistics
    print(f"\nStatistics for {img_path.name}:")
    print(f"  Missed detections: {len(missed)}")
    print(f"  New Detections: {len(false_positive)}")
    if len(false_positive) > 0:
        print(f"  New Detection avg confidence: {sum(fp_confidences)/len(fp_confidences):.3f}")
        print(f"  New Detection confidence range: {min(fp_confidences):.3f} - {max(fp_confidences):.3f}")
    return analysis_image_path

def debug_point_box_matching(image_files, output_path, predictions, img_idx=0, max_distance=100):
    """Debug version that shows detailed matching process"""
    if img_idx >= len(image_files):
        print(f"Image index {img_idx} is out of range (max: {len(image_files)-1})")
        return
        
    img_path = image_files[img_idx]
    print(f"Debugging image: {img_path}")
    
    if str(img_path) not in predictions:
        print(f"No predictions found for {img_path}")
        return
    
    # Get predictions
    pred_boxes = predictions[str(img_path)]
    
    # Create output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = Path(output_path) / f"debug_{timestamp}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    pred_path = debug_dir / f"{img_path.stem}_pred.txt"
    with open(pred_path, 'w') as f:
        for box in pred_boxes:
            box_coords = ' '.join(map(str, box.tolist()))
            f.write(f"{box_coords}\n")
    
    # Get truth path
    truth_path = Path(img_path.parent.parent) / 'labels' / f"{img_path.stem}.txt"
    
    # Load image and boxes manually for debugging
    if img_path.suffix.lower() in ['.tif', '.tiff']:
        with rasterio.open(img_path) as src:
            image = src.read().transpose(1, 2, 0)
            if image.shape[2] >= 3:
                image = image[:, :, [2, 1, 0]]  # RGB to BGR
            height, width = image.shape[:2]
            is_geotiff = True
    else:
        image = cv2.imread(str(img_path))
        height, width = image.shape[:2]
        is_geotiff = False
    
    # Load boxes
    pred_boxes = load_boxes(pred_path)
    truth_boxes_raw = load_boxes(truth_path)
    
    # Convert YOLO format
    truth_boxes = []
    for box in truth_boxes_raw:
        x1 = int((box[0] - box[2]/2) * width)
        y1 = int((box[1] - box[3]/2) * height)
        x2 = int((box[0] + box[2]/2) * width)
        y2 = int((box[1] + box[3]/2) * height)
        truth_boxes.append([x1, y1, x2, y2])
    
    # Create debug images
    base_image = image.copy()
    
    # Draw all truth boxes in red
    truth_img = base_image.copy()
    for i, box in enumerate(truth_boxes):
        cv2.rectangle(truth_img, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     (0, 0, 255), 2)
        # Add box ID
        cv2.putText(truth_img, f"T{i}", (int(box[0]), int(box[1])-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw all predictions in blue
    pred_img = base_image.copy()
    for i, box in enumerate(pred_boxes):
        cv2.rectangle(pred_img, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     (255, 0, 0), 2)
        # Add box ID
        cv2.putText(pred_img, f"P{i}", (int(box[0]), int(box[1])-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Save debug images
    cv2.imwrite(str(debug_dir / f"{img_path.stem}_truth.jpg"), truth_img)
    cv2.imwrite(str(debug_dir / f"{img_path.stem}_pred.jpg"), pred_img)
    
    # Check for points inside prediction boxes
    problem_cases = []
    overlap_img = base_image.copy()
    
    # Identify potential point boxes (small ones)
    point_indices = []
    box_indices = []
    
    # Classify truth boxes as points or boxes based on size
    for i, box in enumerate(truth_boxes):
        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height
        
        if area < 900:  # Area threshold for points
            point_indices.append(i)
        else:
            box_indices.append(i)
    
    print(f"\nFound {len(point_indices)} point annotations and {len(box_indices)} box annotations")
    
    # Check for points inside predictions
    for i in point_indices:
        truth_point = truth_boxes[i]
        center_x = (truth_point[0] + truth_point[2]) / 2
        center_y = (truth_point[1] + truth_point[3]) / 2
        
        # Find predictions that contain this point
        containing_preds = []
        for j, pred_box in enumerate(pred_boxes):
            if (pred_box[0] <= center_x <= pred_box[2] and 
                pred_box[1] <= center_y <= pred_box[3]):
                containing_preds.append(j)
                
                # Draw a green line connecting them
                cv2.line(overlap_img, 
                        (int(center_x), int(center_y)),
                        (int((pred_box[0] + pred_box[2])/2), int((pred_box[1] + pred_box[3])/2)),
                        (0, 255, 0), 2)
                
                # Mark prediction box with green outline
                cv2.rectangle(overlap_img, 
                             (int(pred_box[0]), int(pred_box[1])), 
                             (int(pred_box[2]), int(pred_box[3])), 
                             (0, 255, 0), 2)
                
                # Mark the point with green
                cv2.rectangle(overlap_img, 
                             (int(truth_point[0]), int(truth_point[1])), 
                             (int(truth_point[2]), int(truth_point[3])), 
                             (0, 255, 0), 2)
        
        if not containing_preds:
            # This point is not inside any prediction - problematic!
            problem_cases.append(f"Point T{i} is not inside any prediction")
            cv2.rectangle(overlap_img, 
                         (int(truth_point[0]), int(truth_point[1])), 
                         (int(truth_point[2]), int(truth_point[3])), 
                         (0, 0, 255), 2)  # Red = missed
    
    # Save and show overlap image
    cv2.imwrite(str(debug_dir / f"{img_path.stem}_overlap.jpg"), overlap_img)
    print("\nShowing point-prediction relationships (green lines = match, red = problem):")

    
    # Print problem cases
    if problem_cases:
        print("\nPotential issues identified:")
        for case in problem_cases:
            print(f" - {case}")
    else:
        print("\nNo obvious point-box matching issues found.")
    
    # Now run the regular analysis for comparison
    print("\nRunning standard analysis for comparison:")
    analyze_single_image(predictions, img_idx, True, max_distance)
    
    return debug_dir