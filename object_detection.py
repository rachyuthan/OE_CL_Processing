from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
import pyproj
from shapely.geometry import box as shp_box, Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform
import rasterio
import json
import geopandas as gpd
import os




# =============================================================
# Functions for loading models
# =============================================================

def load_models(model_type, model_version, folds=5):
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
        model = YOLO("./Maxar_skysat_combined/Maxar_images-skysat_combined3/weights/best.pt") #change to correct directory
        return [model]  # Return as list for consistent handling
    elif model_type == 'kfolds':
        k_fold_models = []
        model_dir = Path(f"./new_weights/k_folds_cross_val_{model_version}")
        for k in range(folds):
            model_path = model_dir / f"split_{k+1}" / "weights" / "best.pt"
            if model_path.exists():
                k_fold_models.append(YOLO(model_path))
        return k_fold_models
    else:  # rcnn
        return None  # RCNN doesn't need model loading in the same way
    
# =============================================================
# Functions for rotation-based augmentation
# =============================================================

def rotate_image(image, angle):
    """
    Rotate image by specified angle (90, 180, 270 degrees).
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    angle : int
        Rotation angle (90, 180, or 270)
    
    Returns:
    --------
    rotated : np.ndarray
        Rotated image
    """
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image

def rotate_boxes_back(boxes, angle, orig_width, orig_height):
    """
    Rotate bounding boxes back to original orientation.
    
    Parameters:
    -----------
    boxes : np.ndarray
        Array of bounding boxes [x1, y1, x2, y2] in ROTATED image coordinates
    angle : int
        Rotation angle used (90, 180, or 270)
    orig_width : int
        Original image width (before rotation)
    orig_height : int
        Original image height (before rotation)
    
    Returns:
    --------
    rotated_boxes : np.ndarray
        Boxes transformed back to original coordinate system
        
    Note:
    -----
    For 90/270 rotations, the rotated image has swapped dimensions:
    - Original: orig_width × orig_height
    - After 90° CW: orig_height × orig_width
    The transformation accounts for this dimension swap.
    """
    if len(boxes) == 0:
        return boxes
    
    rotated_boxes = []
    
    # Calculate rotated image dimensions
    if angle in [90, 270]:
        rot_width = orig_height
        rot_height = orig_width
    else:
        rot_width = orig_width
        rot_height = orig_height
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        if angle == 90:
            # When image was rotated 90° CW, rotate coords 90° CCW
            # Rotated image dimensions: rot_width=orig_height, rot_height=orig_width
            # To reverse: (x',y') in rotated -> (x,y) in original
            new_x1 = y1
            new_y1 = rot_width - x2  # rot_width = orig_height
            new_x2 = y2
            new_y2 = rot_width - x1
        elif angle == 180:
            # When rotated 180°, reverse both axes
            new_x1 = rot_width - x2
            new_y1 = rot_height - y2
            new_x2 = rot_width - x1
            new_y2 = rot_height - y1
        elif angle == 270:
            # When image was rotated 270° CW (90° CCW), rotate coords 90° CW
            # Rotated image dimensions: rot_width=orig_height, rot_height=orig_width
            new_x1 = rot_height - y2  # rot_height = orig_width
            new_y1 = x1
            new_x2 = rot_height - y1
            new_y2 = x2
        else:
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
        
        # Ensure x1 < x2 and y1 < y2
        final_box = [
            min(new_x1, new_x2),
            min(new_y1, new_y2),
            max(new_x1, new_x2),
            max(new_y1, new_y2)
        ]
        
        rotated_boxes.append(final_box)
    
    return np.array(rotated_boxes)

def rotate_valid_area_geometry(valid_area_gdf, angle, orig_width, orig_height, transform, image_crs):
    """
    Rotate valid area geometry to match rotated image coordinates.
    
    Parameters:
    -----------
    valid_area_gdf : GeoDataFrame
        Original valid area geometry in geographic coordinates
    angle : int
        Rotation angle (90, 180, or 270)
    orig_width : int
        Original image width in pixels
    orig_height : int
        Original image height in pixels
    transform : affine.Affine
        Original image geotransform
    image_crs : CRS
        Original image CRS
    
    Returns:
    --------
    rotated_gdf : GeoDataFrame
        Valid area geometry transformed to match rotated image
    rotated_transform : affine.Affine
        Geotransform for rotated image
    rotated_width : int
        Width of rotated image
    rotated_height : int
        Height of rotated image
    """
    from affine import Affine
    from shapely.affinity import affine_transform
    
    if angle == 0 or valid_area_gdf is None:
        return valid_area_gdf, transform, orig_width, orig_height
    
    # Calculate rotated image dimensions
    if angle in [90, 270]:
        rotated_width = orig_height
        rotated_height = orig_width
    else:  # 180
        rotated_width = orig_width
        rotated_height = orig_height
    
    # Get the original image bounds in geographic coordinates
    # Top-left corner
    tl_x, tl_y = transform * (0, 0)
    # Top-right corner
    tr_x, tr_y = transform * (orig_width, 0)
    # Bottom-left corner
    bl_x, bl_y = transform * (0, orig_height)
    # Bottom-right corner
    br_x, br_y = transform * (orig_width, orig_height)
    
    # Create the new geotransform based on rotation
    if angle == 90:
        # After 90° CW rotation: top-left becomes top-right
        new_origin_x, new_origin_y = tr_x, tr_y
        # Pixel size changes: x goes down (negative), y goes left
        pixel_x_geo = (br_x - tr_x) / rotated_width  # Moving right in rotated = down in original
        pixel_y_geo = (br_y - tr_y) / rotated_width
        pixel_x_change_geo = (tl_x - tr_x) / rotated_height  # Moving down in rotated = left in original  
        pixel_y_change_geo = (tl_y - tr_y) / rotated_height
        
    elif angle == 180:
        # After 180° rotation: top-left becomes bottom-right
        new_origin_x, new_origin_y = br_x, br_y
        # Pixel directions reverse
        pixel_x_geo = (bl_x - br_x) / rotated_width  # Moving right = left in original
        pixel_y_geo = (bl_y - br_y) / rotated_width
        pixel_x_change_geo = (tr_x - br_x) / rotated_height  # Moving down = up in original
        pixel_y_change_geo = (tr_y - br_y) / rotated_height
        
    elif angle == 270:
        # After 270° CW rotation: top-left becomes bottom-left
        new_origin_x, new_origin_y = bl_x, bl_y
        # Pixel size changes
        pixel_x_geo = (tl_x - bl_x) / rotated_width  # Moving right in rotated = up in original
        pixel_y_geo = (tl_y - bl_y) / rotated_width
        pixel_x_change_geo = (br_x - bl_x) / rotated_height  # Moving down in rotated = right in original
        pixel_y_change_geo = (br_y - bl_y) / rotated_height
    
    # Create new affine transform
    rotated_transform = Affine(
        pixel_x_geo, pixel_x_change_geo, new_origin_x,
        pixel_y_geo, pixel_y_change_geo, new_origin_y
    )
    
    # Rotate the valid area geometry to match the rotated image
    # Step 1: Convert valid area from geographic to pixel coordinates (original image)
    # Step 2: Rotate the pixel coordinates
    # Step 3: Convert back to geographic using rotated transform
    
    from shapely.geometry import Polygon, MultiPolygon
    
    def rotate_geometry_pixels(geom, angle, orig_width, orig_height):
        """Rotate geometry in pixel space"""
        if geom.geom_type == 'Polygon':
            # Rotate exterior ring
            exterior_coords = []
            for x, y in geom.exterior.coords:
                if angle == 90:
                    # 90° CW: (x,y) -> (height-y, x)
                    new_x, new_y = orig_height - y, x
                elif angle == 180:
                    # 180°: (x,y) -> (width-x, height-y)
                    new_x, new_y = orig_width - x, orig_height - y
                elif angle == 270:
                    # 270° CW: (x,y) -> (y, width-x)
                    new_x, new_y = y, orig_width - x
                else:
                    new_x, new_y = x, y
                exterior_coords.append((new_x, new_y))
            
            # Rotate interior rings (holes)
            interior_coords = []
            for interior in geom.interiors:
                interior_ring = []
                for x, y in interior.coords:
                    if angle == 90:
                        new_x, new_y = orig_height - y, x
                    elif angle == 180:
                        new_x, new_y = orig_width - x, orig_height - y
                    elif angle == 270:
                        new_x, new_y = y, orig_width - x
                    else:
                        new_x, new_y = x, y
                    interior_ring.append((new_x, new_y))
                interior_coords.append(interior_ring)
            
            return Polygon(exterior_coords, interior_coords if interior_coords else None)
        
        elif geom.geom_type == 'MultiPolygon':
            rotated_polys = [rotate_geometry_pixels(poly, angle, orig_width, orig_height) 
                           for poly in geom.geoms]
            return MultiPolygon(rotated_polys)
        
        return geom
    
    # Convert valid area to pixel space, rotate, then convert back to geographic
    rotated_geometries = []
    for geom in valid_area_gdf.geometry:
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            # First convert to pixel space using original transform
            if geom.geom_type == 'Polygon':
                polys_to_process = [geom]
            else:
                polys_to_process = list(geom.geoms)
            
            rotated_polys = []
            for poly in polys_to_process:
                # Convert exterior to pixels
                exterior_pixels = []
                for x_geo, y_geo in poly.exterior.coords:
                    row, col = rasterio.transform.rowcol(transform, x_geo, y_geo)
                    exterior_pixels.append((col, row))  # (x, y) in pixel space
                
                # Create pixel polygon
                pixel_poly = Polygon(exterior_pixels)
                
                # Rotate in pixel space
                rotated_pixel_poly = rotate_geometry_pixels(pixel_poly, angle, orig_width, orig_height)
                
                # Convert back to geographic using rotated transform
                exterior_geo = []
                for x_pix, y_pix in rotated_pixel_poly.exterior.coords:
                    x_geo, y_geo = rotated_transform * (x_pix, y_pix)
                    exterior_geo.append((x_geo, y_geo))
                
                rotated_polys.append(Polygon(exterior_geo))
            
            if geom.geom_type == 'Polygon':
                rotated_geometries.append(rotated_polys[0])
            else:
                rotated_geometries.append(MultiPolygon(rotated_polys))
        else:
            rotated_geometries.append(geom)
    
    # Create rotated GeoDataFrame
    rotated_gdf = gpd.GeoDataFrame(geometry=rotated_geometries, crs=image_crs)
    
    return rotated_gdf, rotated_transform, rotated_width, rotated_height

# =============================================================
# Functions for generating predictions
# =============================================================

def non_max_suppression(boxes, scores, iou_threshold=0.5, ensemble_boost=False):
    """
    Apply Non-Max Suppression with confidence weighting and ensemble boosting.
    
    Ensemble Boosting Strategy:
    - Single model detection: confidence × 0.7 (penalize)
    - 2 models agree: confidence × 0.9
    - 3+ models agree: confidence × 1.2 (boost)
    
    This helps prioritize detections where multiple models agree while still
    keeping singleton detections (which may be valid but uncertain).
    
    Parameters:
    -----------
    boxes : list
        List of bounding boxes in the format [x1, y1, x2, y2]
    scores : list
        List of confidence scores for each box
    iou_threshold : float
        IoU threshold for suppression (default: 0.5)
    ensemble_boost : bool
        Whether to apply ensemble-based confidence adjustment (default: True)
    
    Returns:
    --------
    final_boxes : list
        List of final bounding boxes after NMS
    final_scores : list
        List of final confidence scores after NMS (adjusted by ensemble agreement)
    """
    if len(boxes) == 0:
            return [], []
        
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort by confidence score
    order = scores.argsort()[::-1]
    
    
    final_boxes = []
    final_scores = []

    while order.size > 0:
        i = order[0]
        
        # Get overlapping boxes
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / ((boxes[order[1:],2] - boxes[order[1:],0]) * 
                            (boxes[order[1:],3] - boxes[order[1:],1]))
        
        # Find overlapping boxes
        inds = np.where(overlap > iou_threshold)[0]
        
        if len(inds) > 0:
            # Multiple models detected overlapping objects
            overlapping_boxes = np.vstack((boxes[i], boxes[order[inds + 1]]))
            overlapping_scores = np.concatenate(([scores[i]], scores[order[inds + 1]]))
            
            # Weighted average based on confidence scores
            weights = overlapping_scores / np.sum(overlapping_scores)
            avg_box = np.sum(overlapping_boxes * weights[:, np.newaxis], axis=0)
            avg_score = np.mean(overlapping_scores)
            
            # Apply ensemble boost based on number of models agreeing
            if ensemble_boost:
                num_models_agreeing = len(overlapping_scores)
                if num_models_agreeing == 1:
                    # Singleton detection (shouldn't happen in this branch, but safety check)
                    boost_factor = 0.7
                elif num_models_agreeing == 2:
                    # Two models agree - slight penalty
                    boost_factor = 0.9
                elif num_models_agreeing >= 3:
                    # Strong consensus - boost confidence
                    boost_factor = 1.2
                else:
                    boost_factor = 1.0
                
                avg_score = min(1.0, avg_score * boost_factor)  # Cap at 1.0
            
            final_boxes.append(avg_box)
            final_scores.append(avg_score)
        else:
            # Single box with no overlap (singleton detection from one model)
            singleton_score = scores[i]
            if ensemble_boost:
                # Penalize singleton detections
                singleton_score = min(1.0, singleton_score * 0.7)
            
            final_boxes.append(boxes[i])
            final_scores.append(singleton_score)
            
        order = np.delete(order, np.concatenate(([0], inds + 1)))
    
    return np.array(final_boxes), np.array(final_scores)

def sliding_window_detection(
    models,
    image_path,
    window_size: int,
    overlap_ratio: float,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    output_dir: str = None
) -> tuple:
    """
    Perform object detection using a sliding window approach with an ensemble of models.
    
    Parameters
    -----------
    models : list
        List of detection models (ensemble from k-fold cross validation)
    image_path : Path
        Path to the image to process
    window_size : int
        Size of the square sliding window
    overlap_ratio : float
        Amount of overlap between adjacent windows (0.0 to 1.0)
    conf_threshold : float
        Confidence threshold for detections
    iou_threshold : float
        IoU threshold for NMS
        
    Returns
    --------
    final_boxes : np.ndarray
        Detection boxes for the image
    final_confidences : np.ndarray
        Confidence scores for each box
    """
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    prediction_dir = output_dir / f"sw_predictions_{window_size}_{overlap_ratio}"
    prediction_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup paths for saving predictions
    pred_path = prediction_dir / f"{image_path.stem}_pred.npy"
    conf_path = prediction_dir / f"{image_path.stem}_conf.npy"
    
    # Get image dimensions
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    
    # Calculate step size based on window_size and overlap_ratio
    step_size = int(window_size * (1 - overlap_ratio))
    
    # Calculate total number of windows
    y_positions = list(range(0, height, step_size))
    x_positions = list(range(0, width, step_size))
    total_windows = len(y_positions) * len(x_positions)
    
    # Lists to store all detections for this image
    all_boxes = []
    all_scores = []
    
    # Create single progress bar for all windows
    pbar = tqdm(total=total_windows, desc="Processing windows", unit="window")
    
    # Slide window across the image
    y_border = False
    for y in y_positions:
        x_border = False
        for x in x_positions:
            # Extract the window
            
            
            if x + window_size > width:
                x = width - window_size
                x_border = True
            if y + window_size > height:
                y = height - window_size
                y_border = True
            window = image[y:y+window_size, x:x+window_size]
            
            # Process each model in the ensemble
            pred_boxes = []
            pred_confidences = []
            for model in models:
                # Run detection on the window
                results = model.predict(window, conf=conf_threshold, imgsz=window_size, verbose=False)
                
                for result in results:
                    # Extract boxes and confidences
                    pred_boxes.extend(result.boxes.xyxy.cpu().numpy())
                    pred_confidences.extend(result.boxes.conf.cpu().numpy())
            
            # Apply NMS to window predictions
            boxes, confidences = non_max_suppression(pred_boxes, pred_confidences, iou_threshold, ensemble_boost=True)
                    
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
            
            # Update progress bar
            pbar.update(1)
            
            if x_border:
                break
        if y_border:
            break
    
    # Close progress bar
    pbar.close()
    
    # Apply NMS if there are any detections
    final_boxes, final_confidences = non_max_suppression(all_boxes, all_scores, iou_threshold)
    
    # Save predictions
    np.save(pred_path, final_boxes)
    np.save(conf_path, final_confidences)
    
    return final_boxes, final_confidences

def sliding_window_detection_test(
    models,
    image_path,  # Single image path
    window_size: int,
    overlap_ratio: float,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    output_dir: str = None,
    debug: bool = False,
    valid_area_path: Path = None  # Optional: explicit path to valid area GeoJSON
) -> tuple:
    """
    Perform object detection using a sliding window approach with an ensemble of models.
    
    Parameters
    -----------
    models : list
        List of detection models (ensemble from k-fold cross validation)
    image_path : Path
        Path to the image to process
    window_size : int
        Size of the square sliding window
    overlap_ratio : float
        Amount of overlap between adjacent windows (0.0 to 1.0)
    conf_threshold : float
        Confidence threshold for detections
    iou_threshold : float
        IoU threshold for NMS
        
    Returns
    --------
    final_boxes : np.ndarray
        Detection boxes for the image
    final_confidences : np.ndarray
        Confidence scores for each box
    """
    
    
    # Load valid area polygon
    valid_area_gdf = None
    
    # If explicit valid area path provided, use it
    if valid_area_path and valid_area_path.exists():
        valid_area_gdf = gpd.read_file(valid_area_path)
    else:
        # Otherwise, search in default directory
        valid_areas_path = Path('/cephfs/work/rithvik/datasets/datasets/BHE/valid_areas/')
        found_valid_path = None
        
        for valid_path in valid_areas_path.glob("*.json"):
            if image_path.stem in valid_path.stem:
                found_valid_path = valid_path
                break
        
        if found_valid_path and found_valid_path.exists():
            valid_area_gdf = gpd.read_file(found_valid_path)
    
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    prediction_dir = output_dir / f"sw_predictions_{window_size}_{overlap_ratio}"
    prediction_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup paths for saving predictions
    pred_path = prediction_dir / f"{image_path.stem}_pred.npy"
    conf_path = prediction_dir / f"{image_path.stem}_conf.npy"
    
    # Delete cached predictions if they exist to force regeneration
    if pred_path.exists() and conf_path.exists():
        print(f"Deleting cached predictions from: {prediction_dir}")
        os.remove(pred_path)
        os.remove(conf_path)
    
    # Get image dimensions and geotransform
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    
    # Get geotransform for coordinate conversion
    with rasterio.open(image_path) as src:
        transform = src.transform
        image_crs = src.crs
    
    # Convert valid area to pixel coordinates if available
    valid_area_pixel_gdf = None
    if valid_area_gdf is not None:
        # Reproject valid area to image CRS if needed
        if valid_area_gdf.crs != image_crs:
            valid_area_gdf = valid_area_gdf.to_crs(image_crs)
        
        # Convert geometries from geographic to pixel coordinates
        def geo_to_pixel_geometry(geom, transform):
            """Transform geometry from geographic to pixel coordinates"""
            
            
            if geom.geom_type == 'Polygon':
                # Transform exterior ring
                exterior_coords = []
                for x_geo, y_geo in geom.exterior.coords:
                    row, col = rasterio.transform.rowcol(transform, x_geo, y_geo)
                    exterior_coords.append((col, row))  # (x, y) in pixel space
                
                # Transform interior rings (holes)
                interior_coords = []
                for interior in geom.interiors:
                    interior_ring = []
                    for x_geo, y_geo in interior.coords:
                        row, col = rasterio.transform.rowcol(transform, x_geo, y_geo)
                        interior_ring.append((col, row))
                    interior_coords.append(interior_ring)
                
                return Polygon(exterior_coords, interior_coords if interior_coords else None)
            
            elif geom.geom_type == 'MultiPolygon':
                polygons = [geo_to_pixel_geometry(poly, transform) for poly in geom.geoms]
                return MultiPolygon(polygons)
            
            else:
                return geom
        
        # Transform all geometries to pixel space
        pixel_geometries = [geo_to_pixel_geometry(geom, transform) for geom in valid_area_gdf.geometry]
        valid_area_pixel_gdf = gpd.GeoDataFrame(geometry=pixel_geometries, crs=None)  # No CRS for pixel coords

    # Calculate step size based on window_size and overlap_ratio
    step_size = int(window_size * (1 - overlap_ratio))
    
    # Calculate positions (note: actual number may be less due to border logic)
    y_positions = list(range(0, height, step_size))
    x_positions = list(range(0, width, step_size))
    
    # Lists to store all detections for this image
    all_boxes = []
    all_scores = []
    
    # Create progress bar without fixed total (will update as we go)
    pbar = tqdm(desc="Processing windows", unit="window")
    
    # Slide window across the image
    y_border = False
    for y in y_positions:
        x_border = False
        for x in x_positions:
            # Extract the window
            
            
            if x + window_size > width:
                x = width - window_size
                x_border = True
            if y + window_size > height:
                y = height - window_size
                y_border = True
            
            # Check if window overlaps with valid area (in pixel space)
            if valid_area_pixel_gdf is not None:
                # Create window bounding box in pixel coordinates
                window_bbox = shp_box(x, y, x + window_size, y + window_size)
                
                # Check if window intersects with any valid area polygon (in pixel space)
                intersects = valid_area_pixel_gdf.intersects(window_bbox).any()
                
                if not intersects:
                    pbar.update(1)
                    if x_border:
                        break
                    continue  # Skip this window
            
            window = image[y:y+window_size, x:x+window_size]
            
            # Process each model in the ensemble
            pred_boxes = []
            pred_confidences = []
            for model in models:
                # Run detection on the window
                results = model.predict(window, conf=conf_threshold, imgsz=window_size, verbose=False)
                
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
            
            # Update progress bar
            pbar.update(1)
            
            if x_border:
                break
        if y_border:
            break
    
    # Close progress bar
    pbar.close()
    
    
    # Apply NMS if there are any detections
    final_boxes, final_confidences = non_max_suppression(all_boxes, all_scores, iou_threshold)
    
    # Save predictions
    np.save(pred_path, final_boxes)
    np.save(conf_path, final_confidences)
    
    return final_boxes, final_confidences

def sliding_window_detection_with_rotation(
    models,
    image_path,
    window_size: int,
    overlap_ratio: float,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    output_dir: str = None,
    debug: bool = False,
    rotations: list = [0, 90, 180, 270]
) -> tuple:
    """
    Perform object detection with rotation-based test-time augmentation.
    Uses sliding_window_detection_test with valid area filtering for each rotation.
    
    **IMPORTANT**: Returns boxes in PIXEL COORDINATES of the ORIGINAL image orientation.
    These pixel coordinates can be converted to geographic coordinates using the
    ORIGINAL image's geotransform (via predictions_to_geojson or similar functions).
    
    Workflow:
    1. For each rotation (0°, 90°, 180°, 270°):
       - Rotate image and valid area mask
       - Run detection in rotated pixel space
       - Transform boxes back to original pixel space
    2. Combine all boxes (now all in original pixel coordinates)
    3. Apply NMS with ensemble boosting
    4. Return final boxes in original pixel coordinates
    
    Parameters
    -----------
    models : list
        List of detection models (ensemble from k-fold cross validation)
    image_path : Path
        Path to the image to process
    window_size : int
        Size of the square sliding window
    overlap_ratio : float
        Amount of overlap between adjacent windows (0.0 to 1.0)
    conf_threshold : float
        Confidence threshold for detections
    iou_threshold : float
        IoU threshold for NMS
    output_dir : str
        Directory to save predictions
    debug : bool
        Whether to save debug images
    rotations : list
        List of rotation angles to apply (default: [0, 90, 180, 270])
        
    Returns
    --------
    final_boxes : np.ndarray
        Detection boxes in PIXEL COORDINATES of original image [x1, y1, x2, y2]
        Use predictions_to_geojson() to convert to geographic coordinates
    final_confidences : np.ndarray
        Confidence scores for each box
    """
    # Read original image to get dimensions
    original_image = cv2.imread(str(image_path))
    orig_height, orig_width = original_image.shape[:2]
    
    # Get geotransform and CRS for valid area
    with rasterio.open(image_path) as src:
        orig_transform = src.transform
        image_crs = src.crs
    
    # Load valid area for original image (if it exists)
    valid_areas_path = Path('/cephfs/work/rithvik/datasets/datasets/BHE/valid_areas/')
    valid_area_path = None
    
    for valid_path in valid_areas_path.glob("*.json"):
        if image_path.stem in valid_path.stem:
            valid_area_path = valid_path
            break
    
    # Load valid area polygon
    original_valid_area_gdf = None
    if valid_area_path and valid_area_path.exists():
        original_valid_area_gdf = gpd.read_file(valid_area_path)
    
    # Setup output directory
    output_dir = Path(output_dir)
    prediction_dir = output_dir / f"sw_predictions_rot_{window_size}_{overlap_ratio}"
    prediction_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if final predictions already exist and delete them to force regeneration
    final_pred_path = prediction_dir / f"{image_path.stem}_pred.npy"
    final_conf_path = prediction_dir / f"{image_path.stem}_conf.npy"
    
    if final_pred_path.exists() and final_conf_path.exists():
        print(f"Deleting cached rotation predictions from: {prediction_dir}")
        os.remove(final_pred_path)
        os.remove(final_conf_path)
    
    # Collect all detections from all rotations
    all_rotation_boxes = []
    all_rotation_scores = []
    
    # Create temp directory for rotated images
    temp_dir = output_dir / "temp_rotations"
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    for rotation_angle in rotations:
        if rotation_angle == 0:
            # No rotation needed, use original image and valid area
            rotated_image_path = image_path
            # Run detection with original valid area checking
            rot_pred_dir = output_dir / f"temp_rot_0"
            rot_pred_dir.mkdir(exist_ok=True, parents=True)
            
            rot_boxes, rot_confidences = sliding_window_detection_test(
                models,
                rotated_image_path,
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                output_dir=str(rot_pred_dir),
                debug=debug
            )
            # rot_boxes are in pixel coordinates of the ORIGINAL image (0° rotation)
            
        else:
            # Rotate image
            rotated_image = rotate_image(original_image, rotation_angle)
            rot_height, rot_width = rotated_image.shape[:2]
            
            # Get rotated valid area geometry and transform
            rotated_valid_area_gdf, rotated_transform, rot_width_check, rot_height_check = rotate_valid_area_geometry(
                original_valid_area_gdf,
                rotation_angle,
                orig_width,
                orig_height,
                orig_transform,
                image_crs
            )
            
            # Save rotated image as GeoTIFF with rotated transform
            rotated_image_path = temp_dir / f"{image_path.stem}_rot{rotation_angle}.tif"
            
            # Save as GeoTIFF with proper geotransform
            with rasterio.open(
                rotated_image_path,
                'w',
                driver='GTiff',
                height=rot_height,
                width=rot_width,
                count=3,
                dtype=rotated_image.dtype,
                crs=image_crs,
                transform=rotated_transform
            ) as dst:
                # Write RGB channels
                for band in range(3):
                    dst.write(rotated_image[:, :, band], band + 1)
            
            # Run detection on rotated image with rotated valid area
            rot_pred_dir = output_dir / f"temp_rot_{rotation_angle}"
            rot_pred_dir.mkdir(exist_ok=True, parents=True)
            
            # Temporarily save rotated valid area with matchable name
            temp_valid_area_path = None
            if rotated_valid_area_gdf is not None:
                temp_valid_area_path = temp_dir / f"valid_area_rot{rotation_angle}.json"
                rotated_valid_area_gdf.to_file(temp_valid_area_path, driver='GeoJSON')
            
            rot_boxes, rot_confidences = sliding_window_detection_test(
                models,
                rotated_image_path,
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                output_dir=str(rot_pred_dir),
                debug=debug,
                valid_area_path=temp_valid_area_path  # Pass rotated valid area explicitly
            )
            
            # Transform boxes back to original coordinates
            rot_boxes = rotate_boxes_back(rot_boxes, rotation_angle, orig_width, orig_height)
        
        # Collect results (all boxes in original image pixel coordinates)
        if len(rot_boxes) > 0:
            all_rotation_boxes.extend(rot_boxes)
            all_rotation_scores.extend(rot_confidences)
        
    
    # Apply NMS to combine all rotated detections with ensemble boost
    final_boxes, final_confidences = non_max_suppression(
        all_rotation_boxes, 
        all_rotation_scores, 
        iou_threshold=iou_threshold,
        ensemble_boost=True  # Enable ensemble boost for rotation consensus
    )
    
    # Save final predictions (in pixel coordinates of original image)
    np.save(final_pred_path, final_boxes)
    np.save(final_conf_path, final_confidences)
    
    # Clean up temporary prediction directories
    for rotation_angle in rotations:
        rot_pred_dir = output_dir / f"temp_rot_{rotation_angle}"
        if rot_pred_dir.exists():
            import shutil
            shutil.rmtree(rot_pred_dir)
    
    return final_boxes, final_confidences

def sliding_window_detection_different_windows(
    models,
    image_path,
    window_sizes: list,
    overlap_ratio: float,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    output_dir: str = None,
    debug: bool = False,
    use_rotations: bool = True,
    rotations: list = [0, 90, 180, 270]
) -> tuple:
    """
    Perform object detection with multiple window sizes and optional rotation augmentation.
    
    **IMPORTANT**: Returns boxes in PIXEL COORDINATES of the ORIGINAL image orientation.
    These pixel coordinates can be converted to geographic coordinates using the
    ORIGINAL image's geotransform (via predictions_to_geojson or similar functions).
    
    Workflow:
    1. For each window size, call either:
       - sliding_window_detection_with_rotation() if use_rotations=True
       - sliding_window_detection_test() if use_rotations=False
    2. Combine all boxes from all window sizes
    3. Apply final NMS with ensemble boosting
    4. Return final boxes in original pixel coordinates
    
    Parameters
    -----------
    models : list
        List of detection models (ensemble from k-fold cross validation)
    image_path : Path
        Path to the image to process
    window_sizes : list
        List of window sizes to use for sliding window (e.g., [512, 1024, 2048])
    overlap_ratio : float
        Amount of overlap between adjacent windows (0.0 to 1.0)
    conf_threshold : float
        Confidence threshold for detections
    iou_threshold : float
        IoU threshold for NMS
    output_dir : str
        Directory to save predictions
    debug : bool
        Whether to save debug images
    use_rotations : bool
        Whether to use rotation-based test-time augmentation (default: True)
    rotations : list
        List of rotation angles to apply if use_rotations=True (default: [0, 90, 180, 270])
        
    Returns
    --------
    final_boxes : np.ndarray
        Detection boxes in PIXEL COORDINATES of original image [x1, y1, x2, y2]
        Use predictions_to_geojson() to convert to geographic coordinates
    final_confidences : np.ndarray
        Confidence scores for each box
    """
    # Setup output directory
    output_dir = Path(output_dir)
    window_sizes_str = "_".join(map(str, window_sizes))
    rot_suffix = "rot" if use_rotations else "norot"
    prediction_dir = output_dir / f"sw_predictions_multiscale_{rot_suffix}_{window_sizes_str}_{overlap_ratio}"
    prediction_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if final predictions already exist
    final_pred_path = prediction_dir / f"{image_path.stem}_pred.npy"
    final_conf_path = prediction_dir / f"{image_path.stem}_conf.npy"
    
    if final_pred_path.exists() and final_conf_path.exists():
        return np.load(final_pred_path), np.load(final_conf_path)
    
    # Collect all detections from all window sizes
    all_boxes = []
    all_scores = []
    
    # Iterate through each window size and call appropriate function
    for window_size in window_sizes:
        print(f"\nProcessing window size: {window_size}")
        
        if use_rotations:
            # Use rotation-augmented detection
            boxes, confidences = sliding_window_detection_with_rotation(
                models,
                image_path,
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                output_dir=output_dir,
                debug=debug,
                rotations=rotations
            )
        else:
            # Use standard sliding window detection
            print(f"Processing without rotations...")
            boxes, confidences = sliding_window_detection_test(
                models,
                image_path,
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                output_dir=output_dir,
                debug=debug
            )
        
        # Collect results from this window size
        if len(boxes) > 0:
            all_boxes.extend(boxes)
            all_scores.extend(confidences)
    
    # Apply final NMS to combine all detections from all window sizes
    final_boxes, final_confidences = non_max_suppression(
        all_boxes, 
        all_scores, 
        iou_threshold=iou_threshold,
        ensemble_boost=True  # Enable ensemble boost for multi-scale consensus
    )
    
    # Save final predictions (in pixel coordinates of original image)
    np.save(final_pred_path, final_boxes)
    np.save(final_conf_path, final_confidences)
    
    return final_boxes, final_confidences

def generate_predictions(image_files, output_dir, models, conf_threshold=0.6): # Make change for uniformity
    """
    Generate predictions for all images and save them
    
    Parameters
    -----------
    
    image_files : list or Path
        List of image file paths or single image path
    output_dir : str
        Directory to save predictions
    models : list
        List of loaded models for prediction
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

def generate_sw_predictions(image_path, output_dir, models, conf_threshold, window_size=1024, overlap_ratio=0.5, use_rotation=False, use_window_list=False):
    """
    Load previously saved predictions or generate new ones for a single image.
    
    Parameters:
    -----------
    image_path : Path
        Path to the image file
    output_dir : str or Path
        Directory to save predictions
    models : list
        List of models for ensemble prediction
    conf_threshold : float
        Confidence threshold for detections
    window_size : int or list
        Size(s) of sliding window
    overlap_ratio : float
        Overlap ratio for sliding windows
    use_rotation : bool
        Whether to use rotation-based test-time augmentation (default: False)
    
    Returns:
    --------
    sw_predictions : dict
        Dictionary with image path as key and prediction boxes as value
    sw_confidences : dict
        Dictionary with image path as key and confidence scores as value
    """
    if use_rotation and not use_window_list:
        # Use rotation-augmented predictions
        prediction_dir = Path(output_dir) / f"sw_predictions_rot_{window_size}_{overlap_ratio}"
        pred_path = prediction_dir / f"{image_path.stem}_pred.npy"
        conf_path = prediction_dir / f"{image_path.stem}_conf.npy"
        
        # Check if predictions already exist
        if pred_path.exists() and conf_path.exists():
            print(f"Deleting predictions from: {prediction_dir}")
            os.remove(pred_path)
            os.remove(conf_path)
       
        print(f"Starting rotation-augmented prediction...")
        pred_boxes, pred_confidences = sliding_window_detection_with_rotation(
            models,
            image_path,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            conf_threshold=conf_threshold,
            iou_threshold=0.2,
            output_dir=output_dir,
            debug=False,
            rotations=[0, 90, 180, 270]
        )
    elif use_window_list:
        # Use rotation-augmented predictions with different window sizes
        window_sizes_str = "_".join(map(str, window_size))
        rot_suffix = "rot"
        prediction_dir = Path(output_dir) / f"sw_predictions_multiscale_{rot_suffix}_{window_sizes_str}_{overlap_ratio}"
        pred_path = prediction_dir / f"{image_path.stem}_pred.npy"
        conf_path = prediction_dir / f"{image_path.stem}_conf.npy"
        
        # Delete cached predictions if they exist to force regeneration
        if pred_path.exists() and conf_path.exists():
            print(f"Deleting cached predictions from: {prediction_dir}")
            os.remove(pred_path)
            os.remove(conf_path)
        
        print(f"Starting prediction with different window sizes...")
        pred_boxes, pred_confidences = sliding_window_detection_different_windows(
            models,
            image_path,
            window_sizes=window_size,
            overlap_ratio=overlap_ratio,
            conf_threshold=conf_threshold,
            iou_threshold=0.2,
            output_dir=output_dir,
            debug=False,
            use_rotations=use_rotation,
            rotations=[0, 90, 180, 270]
        )
    else:
        # Standard predictions without rotation
        prediction_dir = Path(output_dir) / f"sw_predictions_{window_size}_{overlap_ratio}"
        pred_path = prediction_dir / f"{image_path.stem}_pred.npy"
        conf_path = prediction_dir / f"{image_path.stem}_conf.npy"
        
        # Check if predictions already exist
        if pred_path.exists() and conf_path.exists():
            print(f"Loading predictions from: {prediction_dir}")
            pred_boxes = np.load(pred_path)
            pred_confidences = np.load(conf_path)
        else:
            print(f"Prediction directory does not exist or incomplete, starting prediction...")
            pred_boxes, pred_confidences = sliding_window_detection_test(
                models,
                image_path,
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                conf_threshold=conf_threshold,
                iou_threshold=0.2,
                output_dir=output_dir,
                debug=False
            )
    
    # Return in dictionary format for compatibility with existing code
    sw_predictions = {str(image_path): pred_boxes}
    sw_confidences = {str(image_path): pred_confidences}
    
    return sw_predictions, sw_confidences

def single_image_pred(model_type='kfolds',
    model_version='m',
    image_id=None,
    sliding_window=False,
    conf_threshold=0.4,
    output_dir=None,
    use_rotation=False,
    use_window_list=True
):
    """
    Generate predictions for a single image using the specified model type and version.
    
    Args:
        model_type (str): Type of model ('kfolds', 'yolo', etc.).
        model_version (str): Version of the model to use.
        image_id (str): Image id/path.
        sliding_window (bool): Whether to use sliding window predictions.
        conf_threshold (float): Confidence threshold for predictions.
        output_dir (str): Output directory for predictions.
        use_rotation (bool): Whether to use rotation-based test-time augmentation (default: False).
    
    Returns:
        tuple: Prediction and confidence values.
    """
    models = load_models(model_type=model_type, model_version=model_version, folds=1)
    if not sliding_window:
        prediction, confidence = generate_predictions(
            models,
            image_id,
            conf_threshold=conf_threshold,
        )
    if use_window_list:
        prediction, confidence = generate_sw_predictions(
            image_id,
            output_dir,
            models,
            window_size=[512, 1024],
            conf_threshold=conf_threshold,
            use_rotation=use_rotation,
            use_window_list=use_window_list
        )
    if sliding_window and not use_window_list:
        prediction, confidence = generate_sw_predictions(
            image_id,
            output_dir,
            models,
            window_size=1024,
            conf_threshold=conf_threshold,
            use_rotation=use_rotation,
            use_window_list=False
        )
    
    return prediction, confidence

def predictions_to_geojson(image_file, predictions, confidences, output_path=None):
    """
    Convert predictions to GeoJSON in CRS EPSG:4326.
    
    Args:
        image_file (str or Path): Path to the georeferenced image file (GeoTIFF).
        predictions (dict): Dictionary with image paths as keys and prediction boxes as values.
        output_path (str or Path, optional): Path to save the GeoJSON file. If None, returns GeoJSON dict.
    
    Returns:
        geo_json (dict or None): GeoJSON dictionary if output_path is None, otherwise None (saves to file).
    
    Raises:
        ValueError: If image is not a GeoTIFF or doesn't have geospatial information.
    """
    image_file = Path(image_file)
    
    # Check if image is a GeoTIFF
    if not str(image_file).lower().endswith(('.tif', '.tiff')):
        raise ValueError(f"Image must be a GeoTIFF (.tif or .tiff): {image_file}")
    
    # Get predictions for this image
    image_key = str(image_file)
    if image_key not in predictions:
        raise ValueError(f"No predictions found for image: {image_file}")
    
    pred_boxes = predictions[image_key]
    conf = confidences[image_key] if image_key in confidences else None
    
    # Get geotransform and CRS from GeoTIFF
    try:
        with rasterio.open(image_file) as src:
            transform = src.transform
            crs = src.crs
    except Exception as e:
        raise ValueError(f"Failed to read geospatial information from image: {e}")
    
    if crs is None:
        raise ValueError(f"Image does not have a valid CRS: {image_file}")
    
    # Create transformer to EPSG:4326 if needed
    target_crs = pyproj.CRS.from_epsg(4326)
    need_transform = crs != target_crs
    
    if need_transform:
        transformer = pyproj.Transformer.from_crs(
            crs, target_crs, always_xy=True
        )
    
    # Convert bounding boxes to geographic coordinates
    features = []
    
    for i, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = map(float, box)
        
        # Transform pixel coordinates to geographic coordinates
        ul_x, ul_y = transform * (x1, y1)  # Upper left
        lr_x, lr_y = transform * (x2, y2)  # Lower right
        
        # Create box geometry in original CRS
        building_box = shp_box(ul_x, lr_y, lr_x, ul_y)  # (minx, miny, maxx, maxy)
        
        # Transform to EPSG:4326 if needed
        if need_transform:
            building_box = shapely_transform(transformer.transform, building_box)
        
        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(building_box.exterior.coords)]
            },
            "properties": {
                "id": i,
                "source_image": str(image_file.name),
                "confidence": float(conf[i]) if conf is not None else None
            }
        }
        features.append(feature)
    
    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "EPSG:4326"
            }
        },
        "features": features
    }
    
    # Save or return
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"GeoJSON saved to: {output_path}")
        return None
    else:
        return geojson


