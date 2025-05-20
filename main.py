from post_processing_tools import *


# Global configuration
CONFIG = {
    "pipeline_path": "/home/rithvik/BHE/pipeline.geojson",
    "image_dir": "/cephfs/work/rithvik/datasets/datasets/BHE/2025Q1/images",
    "output_dir": "/home/rithvik/YOLO/BHE/notebook_results/",
    "max_distance": 213.33,
    "model_type": "kfolds",  # Options: "yolo", "kfolds", "rcnn"
    "model_version": "m",    # For kfolds models
    "use_pipeline": True,      # Whether to filter by pipeline distance
    "conf_threshold": 0.01,     # Confidence threshold for predictions
    "fp_confidence": 0.79,  # Confidence threshold for false positives
    "found_shp_file": "/home/rithvik/BHE/bhe_class_location_results_q1_2025/bhe_class_location_results_2025Q1_shapefile/bhe_class_location_results_2025Q1.shp",
    "testing" : True # Flag to run the comparison with the found shapefiles
}


# Create output directory
Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

# Load the models 
models = load_models(model_type=CONFIG["model_type"], model_version=CONFIG["model_version"])
print(f"Loaded {len(models) if models else 0} models for {CONFIG['model_type']}")

# Get all image paths
image_files = get_image_paths(image_dir=CONFIG["image_dir"])


### Change logic to check if predictions already exist ###
# Generate and save predictions (not sliding window)
# predictions, confidences = generate_predictions(models, image_files, CONFIG['output_dir'], CONFIG['model_type'], CONFIG['conf_threshold'])

# # Load saved predictions
# predictions = load_saved_predictions(image_files, CONFIG['output_dir'])

# Generate and save predictions (sliding window)

sw_predictions, sw_confidences = load_saved_predictions(image_files, CONFIG['output_dir'], models=models, conf_threshold=CONFIG['conf_threshold'])




# Sensitivity analysis

fp_confidences = np.arange(0.4, 0.9, 0.01)
confidence_threshold = np.arange(0.01,0.51, 0.05)
sens_fp = False
sens_conf = False
while True:
    display = input("Perform Sensitivity Analysis? (y/n): ").strip().lower()
    if display == 'y':
        display = input("Sensitivity analysis for FP confidence or confidence threshold? (fp/conf): ").strip().lower()
        if display == 'fp':
            sens_fp = True
            break
        elif display == 'conf':
            sens_conf = True
            break
        else:
            print("Invalid input. Please enter 'fp' or 'conf'.")
    elif display == 'n':
        print("Performing normal analysis.")
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")

if not sens_fp and not sens_conf:
    # Perform analysis

    results_dir, metrics = process_images_with_saved_predictions(
        image_files, 
        sw_predictions,
        sw_confidences, # or predictions if not sliding window 
        model_version=CONFIG['model_version'],
        model_type=CONFIG['model_type'],
        output_dir=CONFIG['output_dir'],
        found_shp_file=CONFIG['found_shp_file'],
        use_pipeline=CONFIG['use_pipeline'],
        pipeline_path=CONFIG['pipeline_path'],
        max_distance=CONFIG['max_distance'],
        fp_confidence=CONFIG['fp_confidence'],
        testing=CONFIG['testing']  # Pass the testing flag if comparison to reported
    )
# Perform sensitivity analysis for FP confidence
if sens_fp:
    for fp in fp_confidences:
        print(f"Testing with FP confidence: {fp}", flush=True)
        CONFIG['fp_confidence'] = fp
        results_dir, metrics = process_images_with_saved_predictions(
            image_files, 
            sw_predictions, 
            CONFIG['output_dir'],
            model_version=CONFIG['model_version'],
            model_type=CONFIG['model_type'],
            found_shp_file=CONFIG['found_shp_file'],
            use_pipeline=CONFIG['use_pipeline'],
            pipeline_path=CONFIG['pipeline_path'],
            max_distance=CONFIG['max_distance'],
            fp_confidence=CONFIG['fp_confidence'],
            testing=CONFIG['testing'],  # Pass the testing flag
            save_images=False  # Save images for this test
        )
        print(f"Results saved to: {results_dir}")
        # Define the CSV file path outside the loop
        sensitivity_results_file = Path(CONFIG['output_dir']) / f"sensitivity_analysis_{CONFIG['model_type']}{'-'+CONFIG['model_version'] if CONFIG['model_version'] else ''}.csv"
        
        # Convert metrics dictionary to a DataFrame row
        metrics_df = pd.DataFrame([metrics])
        metrics_df['fp_confidence_threshold'] = fp # Add the current threshold value
        
        # Check if the file exists to decide whether to write the header
        write_header = not sensitivity_results_file.exists()
        
        # Append the metrics to the CSV file
        metrics_df.to_csv(sensitivity_results_file, mode='a', header=write_header, index=False)
        
        print(f"Metrics appended to: {sensitivity_results_file}")
if sens_conf:
    for conf in confidence_threshold:
        print(f"Testing with confidence threshold: {conf}")
        CONFIG['confidence_threshold'] = conf
        sw_predictions, sw_confidences = sliding_window_detection(
            models,
            image_files,
            window_size=1024,
            overlap_ratio=0.5,
            conf_threshold=conf,
            iou_threshold=0.2,
            output_dir=CONFIG['output_dir'],
            debug=False
        )

        results_dir, metrics = process_images_with_saved_predictions(
            image_files, 
            sw_predictions, 
            CONFIG['output_dir'],
            use_pipeline=CONFIG['use_pipeline'],
            max_distance=CONFIG['max_distance'],
            testing=CONFIG['testing'],  # Pass the testing flag
            save_images=False  # Save images for this test
        )
        print(f"Results saved to: {results_dir}")
        # Define the CSV file path outside the loop
        sensitivity_results_file = Path(CONFIG['output_dir']) / f"sensitivity_analysis_{CONFIG['model_type']}{'-'+CONFIG['model_version'] if CONFIG['model_version'] else ''}_conf_threshold.csv"
        
        # Convert metrics dictionary to a DataFrame row
        metrics_df = pd.DataFrame([metrics])
        metrics_df['confidence_threshold'] = conf # Add the current threshold value
        
        # Check if the file exists to decide whether to write the header
        write_header = not sensitivity_results_file.exists()
        
        # Append the metrics to the CSV file
        metrics_df.to_csv(sensitivity_results_file, mode='a', header=write_header, index=False)

# Usage example of filtering specific items from shapefile
# query = "Playground"
# print(f"Number of {query} in the shapefile:")
# extent_shapefile_path = "/home/rithvik/BHE/gas_transmission/gas_transmission.gt_building_exi_extent.shp"
# field_count = count_type(extent_shapefile_path, query)
# print(field_count)

