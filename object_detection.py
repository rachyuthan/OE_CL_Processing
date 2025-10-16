from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
# =============================================================
# Functions for loading models
# =============================================================

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
        model_dir = Path(f"./new_weights/k_folds_cross_val_{model_version}") #change to correct directory
        print(f"Loading k-fold models from {model_dir}")
        for k in range(5):
            model_path = model_dir / f"split_{k+1}" / "weights" / "best.pt"
            if model_path.exists():
                k_fold_models.append(YOLO(model_path))
        return k_fold_models
    else:  # rcnn
        return None  # RCNN doesn't need model loading in the same way
    
# =============================================================
# Functions for generating predictions
# =============================================================

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Max Suppression with confidence weighting to boxes 
    
    Parameters:
    -----------
    boxes : list
        List of bounding boxes in the format [x1, y1, x2, y2]
    scores : list
        List of confidence scores for each box
    iou_threshold : float
        IoU threshold for suppression
    Returns:
    --------
    final_boxes : list
        List of final bounding boxes after NMS
    final_scores : list
        List of final confidence scores after NMS
    """
    if len(boxes) == 0:
            return [], []
        
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort by confidence score
    order = scores.argsort()[::-1]
    
    keep = []
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
            # Average the boxes and confidences
            overlapping_boxes = np.vstack((boxes[i], boxes[order[inds + 1]]))
            overlapping_scores = np.concatenate(([scores[i]], scores[order[inds + 1]]))
            
            # Weighted average based on confidence scores
            weights = overlapping_scores / np.sum(overlapping_scores)
            avg_box = np.sum(overlapping_boxes * weights[:, np.newaxis], axis=0)
            avg_score = np.mean(overlapping_scores)  # Simple average for confidence
            
            final_boxes.append(avg_box)
            final_scores.append(avg_score)
        else:
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            
        order = np.delete(order, np.concatenate(([0], inds + 1)))
    
    return np.array(final_boxes), np.array(final_scores)

def sliding_window_detection(
    models,
    image_path,  # Single image path
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
    debug : bool
        Whether to save debug images
        
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
    
    # Create debug directory if debug is enabled
    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True, parents=True)
    
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
            # Save debug window if requested
            if debug:
                window_path = debug_dir / f"window{pbar.n}.jpg"
                cv2.imwrite(str(window_path), window)
            
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

def sliding_window_detection_test(
    models,
    image_path,  # Single image path
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
    debug : bool
        Whether to save debug images
        
    Returns
    --------
    final_boxes : np.ndarray
        Detection boxes for the image
    final_confidences : np.ndarray
        Confidence scores for each box
    """
    # Load valid area mask
    import json
    import geopandas as gpd
    from shapely.geometry import box as shapely_box
    import rasterio
    
    valid_areas_path = Path('/cephfs/work/rithvik/datasets/datasets/BHE/valid_areas/')
    valid_area_path = None
    
    for valid_path in valid_areas_path.glob("*.json"):
        if image_path.stem in valid_path.stem:
            print(f"Found valid area mask: {valid_path}")
            valid_area_path = valid_path
            break
    
    # Load valid area polygon
    valid_area_gdf = None
    if valid_area_path and valid_area_path.exists():
        valid_area_gdf = gpd.read_file(valid_area_path)
        print(f"Loaded valid area with CRS: {valid_area_gdf.crs}")
    else:
        print("Warning: No valid area mask found, processing entire image")
    
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    prediction_dir = output_dir / f"sw_predictions_{window_size}_{overlap_ratio}"
    prediction_dir.mkdir(exist_ok=True, parents=True)
    
    # Create debug directory if debug is enabled
    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup paths for saving predictions
    pred_path = prediction_dir / f"{image_path.stem}_pred.npy"
    conf_path = prediction_dir / f"{image_path.stem}_conf.npy"
    
    # Get image dimensions and geotransform
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    
    # Get geotransform for coordinate conversion
    with rasterio.open(image_path) as src:
        transform = src.transform
        image_crs = src.crs
    
    # Calculate step size based on window_size and overlap_ratio
    step_size = int(window_size * (1 - overlap_ratio))
    
    # Calculate positions (note: actual number may be less due to border logic)
    y_positions = list(range(0, height, step_size))
    x_positions = list(range(0, width, step_size))
    
    # Lists to store all detections for this image
    all_boxes = []
    all_scores = []
    
    # Counters for statistics
    skipped_windows = 0
    processed_windows = 0
    total_windows_checked = 0
    
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
            
            # Count this window
            total_windows_checked += 1
            
            # Check if window overlaps with valid area
            if valid_area_gdf is not None:
                # Convert window corners to geographic coordinates
                x1_geo, y1_geo = transform * (x, y)  # Upper left
                x2_geo, y2_geo = transform * (x + window_size, y + window_size)  # Lower right
                
                # Create window bounding box in geographic coordinates
                window_bbox = shapely_box(min(x1_geo, x2_geo), min(y1_geo, y2_geo), 
                                         max(x1_geo, x2_geo), max(y1_geo, y2_geo))
                window_gdf = gpd.GeoDataFrame([1], geometry=[window_bbox], crs=image_crs)
                
                # Reproject to match valid area CRS if needed
                if window_gdf.crs != valid_area_gdf.crs:
                    window_gdf = window_gdf.to_crs(valid_area_gdf.crs)
                
                # Check if window intersects with any valid area polygon
                intersects = window_gdf.intersects(valid_area_gdf.unary_union).any()
                
                if not intersects:
                    skipped_windows += 1
                    pbar.update(1)
                    if x_border:
                        break
                    continue  # Skip this window
            
            processed_windows += 1
            window = image[y:y+window_size, x:x+window_size]
            # Save debug window if requested
            if debug:
                window_path = debug_dir / f"window{pbar.n}.jpg"
                cv2.imwrite(str(window_path), window)
            
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
    
    # Print statistics
    print(f"\n Window processing statistics:")
    print(f"  Total windows checked: {total_windows_checked}")
    print(f"  Processed: {processed_windows}")
    print(f"  Skipped (outside valid area): {skipped_windows}")
    if total_windows_checked > 0:
        print(f"  Efficiency gain: {(skipped_windows/total_windows_checked)*100:.1f}% reduction")
    
    # Apply NMS if there are any detections
    final_boxes, final_confidences = non_max_suppression(all_boxes, all_scores, iou_threshold)
    
    # Save predictions
    np.save(pred_path, final_boxes)
    np.save(conf_path, final_confidences)
    
    return final_boxes, final_confidences

def generate_predictions(models, image_files, output_dir, model_type, conf_threshold): # Make change for uniformity
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

def generate_sw_predictions(image_path, output_dir, models, conf_threshold, window_size=1024, overlap_ratio=0.5):
    """Load previously saved predictions or generate new ones for a single image"""
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
    output_dir=None
):
    """
    Generate predictions for a single image using the specified model type and version.
    
    Args:
        model_version (str): Version of the model to use.
        image_id (str): image id.
        sliding_window (bool): Whether to use sliding window predictions.
        conf_threshold (float): Confidence threshold for predictions.
    
    Returns:
        tuple: Prediction and confidence values.
    """
    models = load_models(model_type=model_type, model_version=model_version)
    if not sliding_window:
        prediction, confidence = generate_predictions(
            models,
            image_id,
            conf_threshold=conf_threshold,
        )
    else:
        prediction, confidence = generate_sw_predictions(
            image_id,
            output_dir,
            models,
            conf_threshold=conf_threshold,
        )
    
    return prediction, confidence



