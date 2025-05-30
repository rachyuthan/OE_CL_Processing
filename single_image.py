"""
File to run inference on a single image. Gives visualization of prediction along 
with confidence scores."""
from post_processing_tools import *
from pathlib import Path
import cv2

CONFIG = {
    "output_dir": "./OE_CL_Processing/single_image/",
}

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

def single_image_pred(model_type='kfolds',
    model_version='m',
    image_path=None,
    sliding_window=False,
    conf_threshold=0.4,
):
    """
    Generate predictions for a single image using the specified model type and version.
    
    Args:
        model_type (str): Type of model to use ('kfolds', 'yolo', 'rcnn').
        model_version (str): Version of the model to use.
        image_path (str): Path to the image file.
        sliding_window (bool): Whether to use sliding window predictions.
        conf_threshold (float): Confidence threshold for predictions.
    
    Returns:
        tuple: Prediction and confidence values.
    """
    models = load_models(model_type=model_type, model_version=model_version)
    if not sliding_window:
        prediction, confidence = generate_predictions(
            models,
            image_path,
            CONFIG['output_dir'],
            model_type=model_type,
            conf_threshold=conf_threshold,
        )
    else:
        prediction, confidence = load_saved_predictions(
            models,
            image_path,
            CONFIG['output_dir'],
            model_type=model_type,
            conf_threshold=conf_threshold,
        )
    
    return prediction, confidence

def visualize_predictions(image_path, predictions, confidences, output_dir):
    """
    Visualize predictions on the image and save the output.
    
    Args:
        image_path (str): Path to the image file.
        predictions (list): List of predictions.
        confidences (list): List of confidence scores.
        output_dir (str): Directory to save the output image.
    """
    img = cv2.imread(image_path)
    pred_boxes = predictions[str(image_path)]
    print(pred_boxes)
    pred_confidences = confidences[str(image_path)] if str(image_path) in confidences else None
    
    for i, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = box.astype(int)  # Convert to integers for cv2
        
        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add confidence score if available
        if pred_confidences is not None and i < len(pred_confidences):
            confidence_text = f'{pred_confidences[i]:.2f}'
            cv2.putText(img, confidence_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

    output_path = Path(output_dir) / f'{Path(image_path).stem}_visualized.png'
    cv2.imwrite(str(output_path), img)

def main():
    """
    Main function to run the single image prediction.
    """
    # Define the image path
    image_path = Path('/cephfs/work/rithvik/datasets/datasets/BHE/2025Q1/images/18065975_43.tif')

    # Run the prediction
    prediction, confidence = single_image_pred(
        model_type='kfolds',
        model_version='m',
        image_path=image_path,
        sliding_window=False,
        conf_threshold=0.4,
    )
    print(prediction)
    # Visualize the predictions
    visualize_predictions(
        image_path=image_path,
        predictions=prediction,
        confidences=confidence,
        output_dir=CONFIG['output_dir'],
    )
    print(f"Predictions saved to {CONFIG['output_dir']}")
    
if __name__ == "__main__":
    main()