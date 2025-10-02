from post_processing_tools import *
import random
import pyproj

# Global configuration
CONFIG = {
    "pipeline_path": "/cephfs/work/rithvik/OE_CL_shps/pipeline.geojson",
    "image_dir": "/cephfs/work/rithvik/datasets/datasets/BHE/2024_GBA/autosplit_test.txt",
    "output_dir": "/home/rithvik/YOLO/BHE/comparison_new/",
    "max_distance": 213.33,
    "model_type": "kfolds",  # Options: "yolo", "kfolds", "rcnn"
    "model_version": "m",    # For kfolds models
    "use_pipeline": True,      # Whether to filter by pipeline distance
    "conf_threshold": 0.01,     # Confidence threshold for predictions
    "fp_confidence": 0.50,  # Confidence threshold for false positives
    "found_shp_file": "/cephfs/work/rithvik/OE_CL_shps/bhe_class_location_results_q1_2025/bhe_class_location_results_2025Q1_shapefile/bhe_class_location_results_2025Q1.shp",
    "testing" : True # Flag to run the comparison with the found shapefiles
}


# Create output directory
Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

# Load the models 
models = load_models(model_type=CONFIG["model_type"], model_version=CONFIG["model_version"])
print(f"Loaded {len(models) if models else 0} models for {CONFIG['model_type']}")

# Get all image paths
image_files = get_image_paths(image_source=CONFIG["image_dir"])


### Change logic to check if predictions already exist ###
# Generate and save predictions (not sliding window)
# predictions, confidences = generate_predictions(models, image_files, CONFIG['output_dir'], CONFIG['model_type'], CONFIG['conf_threshold'])

# # Load saved predictions
# predictions = load_saved_predictions(image_files, CONFIG['output_dir'])

# Generate and save predictions (sliding window)

sw_predictions, sw_confidences = generate_sw_predictions(image_files, CONFIG['output_dir'], models=models, conf_threshold=CONFIG['conf_threshold'])

# === SHAPEFILE FILTERING SETUP ===
preloaded_buildings = None  # Initialize to None

while True:
    display = input("Filter baseline by shapefile types? (y/n): ").strip().lower()
    if display == 'y':
        exclude_types = get_user_filter_preferences()
        sample_crs = pyproj.CRS.from_epsg(4326)
        preloaded_buildings = preload_and_filter_shapefiles(sample_crs, exclude_types=exclude_types)
        break
    elif display == 'n':
        print("No filtering will be applied to shapefiles.")
        preloaded_buildings = None
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")

while True:
    display = input("Generate dummy data? (y/n): ").strip().lower()
    if display == 'y':
        generate_dummy_data = True
        break
    elif display == 'n':
        generate_dummy_data = False
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
if generate_dummy_data:
    # Generate dummy data for testing
    print("Generating dummy data...")
    dummy_predictions = random.sample(range(len(image_files)), min(10, len(image_files)))

    all_csv_data = []
    output_dirs_to_cleanup = []
    
    for num in dummy_predictions:
        analysis_image_path = analyze_single_image(image_files, output_dir=CONFIG["output_dir"], pipeline_path=CONFIG["pipeline_path"],
                            predictions=sw_predictions, confidences=sw_confidences, img_idx=num, use_pipeline=CONFIG["use_pipeline"],
                            max_distance=CONFIG["max_distance"], fp_confidence=CONFIG["fp_confidence"], dummy=True, 
                            preloaded_buildings=preloaded_buildings)
        # Find the generated directory for this analysis
        if analysis_image_path:
            analysis_dir = Path(analysis_image_path).parent
            output_dirs_to_cleanup.append(analysis_dir)
            
            # Look for CSV files in this directory
            csv_files = list(analysis_dir.glob("*_combined_detections.csv"))
            
            for csv_file in csv_files:
                try:
                    # Read the CSV and add image identifier
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        all_csv_data.append(df)
                        print(f"Collected data from {csv_file.name}")
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
    
    # Combine all CSV data into one DataFrame
    if all_csv_data:
        combined_df = pd.concat(all_csv_data, ignore_index=True)
        
        # Create combined CSV filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_csv_path = Path(CONFIG["output_dir"]) / f"combined_dummy_analysis_{timestamp}.csv"
        
        # Save the combined CSV
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\nCombined CSV saved to: {combined_csv_path}")
        print(f"Total records: {len(combined_df)}")
        
        # Print summary by type
        if 'type' in combined_df.columns:
            type_counts = combined_df['type'].value_counts()
            print("Summary by detection type:")
            for detection_type, count in type_counts.items():
                print(f"  {detection_type}: {count}")
        
        # Print summary by source image
        if 'source_image' in combined_df.columns:
            image_counts = combined_df['source_image'].value_counts()
            print(f"\nSummary by image ({len(image_counts)} images):")
            for image_name, count in image_counts.items():
                print(f"  {image_name}: {count} detections")
    
    # Clean up individual directories
    print(f"\nCleaning up {len(output_dirs_to_cleanup)} temporary directories...")
    for dir_path in output_dirs_to_cleanup:
        try:
            # Remove all files in the directory
            for file_path in dir_path.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            
            # Remove the directory itself
            dir_path.rmdir()
            print(f"Deleted: {dir_path.name}")
        except Exception as e:
            print(f"Error deleting {dir_path}: {e}")
    
    print("Dummy data analysis complete!")
if not generate_dummy_data:

    while True:
        display = input("Enter debugging mode? (y/n): ").strip().lower()
        if display == 'y':
            debugging = True
            break
        elif display == 'n':
            debugging = False
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    if debugging:
        # Debugging code
        print("Debugging mode is enabled.")

        analyze_single_image(image_files, output_dir=CONFIG["output_dir"], pipeline_path=CONFIG["pipeline_path"], 
                            predictions=sw_predictions, confidences=sw_confidences, img_idx=58, use_pipeline=True, 
                            max_distance=213.33, fp_confidence=0.4, preloaded_buildings=preloaded_buildings)

    elif not debugging:

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
                testing=CONFIG['testing'],  # Pass the testing flag if comparison to reported
                preloaded_buildings=preloaded_buildings  # Pass pre-loaded shapefiles
            )
        # Perform sensitivity analysis for FP confidence
        if sens_fp:
            for fp in fp_confidences:
                print(f"Testing with FP confidence: {fp}", flush=True)
                CONFIG['fp_confidence'] = fp
                results_dir, metrics = process_images_with_saved_predictions(
                    image_files, 
                    sw_predictions,
                    sw_confidences, 
                    output_dir=CONFIG['output_dir'],
                    model_version=CONFIG['model_version'],
                    model_type=CONFIG['model_type'],
                    found_shp_file=CONFIG['found_shp_file'],
                    use_pipeline=CONFIG['use_pipeline'],
                    pipeline_path=CONFIG['pipeline_path'],
                    max_distance=CONFIG['max_distance'],
                    fp_confidence=CONFIG['fp_confidence'],
                    testing=CONFIG['testing'],  # Pass the testing flag
                    save_images=False,  # Save images for this test
                    preloaded_buildings=preloaded_buildings  # Pass pre-loaded shapefiles
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
                    sw_confidences, 
                    output_dir=CONFIG['output_dir'],
                    model_version=CONFIG['model_version'],
                    model_type=CONFIG['model_type'],
                    found_shp_file=CONFIG['found_shp_file'],
                    use_pipeline=CONFIG['use_pipeline'],
                    pipeline_path=CONFIG['pipeline_path'],
                    max_distance=CONFIG['max_distance'],
                    fp_confidence=CONFIG['fp_confidence'],
                    testing=CONFIG['testing'],  # Pass the testing flag
                    save_images=False,  # Save images for this test
                    preloaded_buildings=preloaded_buildings  # Pass pre-loaded shapefiles
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


