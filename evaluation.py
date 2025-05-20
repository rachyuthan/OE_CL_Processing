import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def create_final_histogram(areas_missed, output_dir):
    """
    Create histograms for missed boxes
    Parameters:
    -----------
    areas_missed : list
        List of areas of missed boxes
    output_dir : str
        Directory to save output files
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    if len(areas_missed) > 0:
        # Missed boxes histogram
        bin_edges = range(0, int(max(areas_missed)) + 100, 100)
        ax1.hist(areas_missed, bins=bin_edges, edgecolor='black')
        ax1.set_title(f'Missed Boxes Area Distribution\nTotal: {len(areas_missed)}')
        ax1.set_xlabel('Area (pixels)')
        ax1.set_ylabel('Count')
        ax1.xaxis.set_major_locator(plt.MultipleLocator(1000))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_histogram.png'))
        plt.close()

def create_final_pie_chart(total_missed, total_buildings, output_dir):
    """
    Create pie chart of detection results, treating partial detections as correct
    Parameters:
    -----------
    total_missed : int
        Total number of missed boxes
    total_buildings : int
        Total number of buildings
    output_dir : str
        Directory to save output files
    """
    plt.figure(figsize=(8, 8))
    
    # Calculate correct detections (now including partial detections)
    correct_matches = total_buildings - total_missed
    
    # Create pie chart with only two categories: Correct (including partial) and Missed
    labels = ['Correct', 'Missed']
    sizes = [correct_matches, total_missed]
    colors = ['green', 'red']
    
    # Add detailed data in the title
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            textprops={'fontsize': 14, 'color': 'black', 'weight': 'bold'})
    plt.title(f'Overall Building Detection Analysis\n' +
              f'Total Buildings: {total_buildings}\n' +
              f'Perfect Matches: {correct_matches}') 

    plt.savefig(os.path.join(output_dir, 'final_pie_chart.png'))
    plt.close()
    
def calculate_box_metrics(image_path, boxes):
    """
    Calculate box metrics including area, distance from edges, and average brightness
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    boxes : list
        List of bounding boxes in the format [x1, y1, x2, y2]
    Returns:
    --------
    metrics : list
        List of dictionaries containing metrics for each box
    """
    # Skip processing if no boxes
    if not boxes:
        return []
        
    # Read image with error handling
    img = cv2.imread(str(image_path))
    
    # Check if image was loaded successfully
    if img is None:
        print(f"Warning: Failed to load image: {image_path}")
        return [{'area': 0, 'min_edge_dist': 0, 'avg_brightness': 0} for _ in boxes]
    
    img_height, img_width = img.shape[:2]
    
    # Calculate box metrics
    metrics = []
    for box in boxes:
        try:
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width-1))
            y1 = max(0, min(y1, img_height-1))
            x2 = max(x1+1, min(x2, img_width))
            y2 = max(y1+1, min(y2, img_height))
            
            # Box area
            area = (x2 - x1) * (y2 - y1)

            # Distance from edges
            dist_left = x1
            dist_right = img_width - x2
            dist_top = y1
            dist_bottom = img_height - y2
            min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)

            # Calculate brightness with safety check
            box_region = img[y1:y2, x1:x2]
            if box_region.size == 0:
                avg_brightness = 0
            else:
                avg_brightness = np.mean(cv2.cvtColor(box_region, cv2.COLOR_BGR2GRAY))

            metrics.append({
                'area': area,
                'min_edge_dist': min_edge_dist,
                'avg_brightness': avg_brightness
            })
        except Exception as e:
            print(f"Error processing box {box}: {e}")
            metrics.append({'area': 0, 'min_edge_dist': 0, 'avg_brightness': 0})
    
    return metrics

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


