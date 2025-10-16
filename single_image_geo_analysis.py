"""
File to run inference on multiple images. Gives visualization of predictions along 
with confidence scores and baseline comparison results."""
from object_detection import *
from baseline_comparison import *
from post_processing import *
from flagging import *
from pathlib import Path
import cv2
import pandas as pd
from datetime import datetime
import shutil

CONFIG = {
   "output_dir": "./single_image/",
   "pipeline_path": "/cephfs/work/rithvik/OE_CL_shps/FullSystem/EGTS_Full_System.shp",
   # For multiple truth files (e.g., points and polygons), provide a list
   "baseline": [
       "/cephfs/work/rithvik/OE_CL_shps/EGTS Buildings/EGTS Buildings/Building Extent.shp",  # extent path
       "/cephfs/work/rithvik/OE_CL_shps/EGTS Buildings/EGTS Buildings/Building Location.shp",  # location path
   ],
   # Input: either a single image path or a directory containing images
   "input_images": "/cephfs/work/rithvik/datasets/datasets/BHE/geo_imgs_test/",  # Directory or single file path
   # Project metadata for CSV export
   "project_id": "85",
   "model_version": "YOLOv11m",
   "cycle_start": "2025-02-01 00:00:00",
   "cycle_end": "2025-04-01 00:00:00"
}


Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

def is_georeferenced(image_path):
    """
    Check if an image has valid georeferencing information.
    
    Args:
        image_path (str or Path): Path to the image file
        
    Returns:
        bool: True if image has valid CRS and geotransform, False otherwise
    """
    try:
        import rasterio
        with rasterio.open(image_path) as src:
            # Check if CRS is defined
            if src.crs is None:
                return False
            # Check if geotransform is valid (not identity transform)
            transform = src.transform
            if transform == rasterio.Affine.identity():
                return False
            return True
    except Exception as e:
        print(f"Error checking georeferencing: {e}")
        return False
    
def visualize_predictions(image_file, predictions, confidences):
    """
    Visualize predictions on the image and save the output.
    
    Args:
        image_file (str): Path to the image file.
        predictions (list): List of predictions.
        confidences (list): List of confidence scores.
    """
    img = cv2.imread(image_file)
    pred_boxes = predictions[str(image_file)]
    #print(pred_boxes)
    pred_confidences = confidences[str(image_file)] if str(image_file) in confidences else None
    
    for i, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = box.astype(int)  # Convert to integers for cv2
        
        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add confidence score if available
        if pred_confidences is not None and i < len(pred_confidences):
            confidence_text = f'{pred_confidences[i]:.2f}'
            cv2.putText(img, confidence_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    output_path = '{}_visualized.png'.format(Path(image_file).stem)
    cv2.imwrite(str(output_path), img)

def get_image_list(input_path):
    """
    Get list of images from either a single file or directory.
    
    Args:
        input_path (str or Path): Path to single image or directory
        
    Returns:
        list: List of image paths
    """
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single image
        if input_path.suffix.lower() in ['.tif', '.tiff']:
            return [input_path]
        else:
            print(f"Warning: {input_path} is not a recognized image format")
            return []
    elif input_path.is_dir():
        # Directory of images
        image_extensions = ['.tif', '.tiff']
        images = []
        for ext in image_extensions:
            images.extend(input_path.glob(f'*{ext}'))
            images.extend(input_path.glob(f'*{ext.upper()}'))
        return sorted(images)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return []
    
def create_building_record(box, transform, image_path, building_type, config):
    """
    Create a dictionary record for a building (new or removed).
    
    Args:
        box (array): Bounding box in pixel coordinates [x1, y1, x2, y2]
        transform (affine.Affine): Geotransform from image
        image_path (Path): Path to the image file
        building_type (str): 'new' or 'removed'
        config (dict): Configuration dictionary with project metadata
        
    Returns:
        dict: Building record dictionary
    """
    x1, y1, x2, y2 = map(float, box)
    
    # Convert to geographic coordinates
    ul_x, ul_y = transform * (x1, y1)  # Upper Left
    lr_x, lr_y = transform * (x2, y2)  # Lower Right
    
    # Mapping of numerical IDs to filenames
    mapping_str = """
    452121     Maxar-50cm_SKYWATCH_25SEP08162820-S3DS_R1C2-200009812612_01_P001_1_px4oFlukr3.tif
    452112     Maxar-50cm_SKYWATCH_25SEP08162820-S3DS_R1C1-200009812612_01_P001_0_BYIuWJ67y9.tif
    452125     Maxar-50cm_SKYWATCH_25SEP08162820-S3DS_R1C3-200009812612_01_P001_2_BRgxfIJ9vh.tif
    452189     Maxar-50cm_SKYWATCH_25SEP08162829-S3DS_R1C2-200009812586_01_P001_1_q4epLnOYnX.tif
    452190     Maxar-50cm_SKYWATCH_25SEP08162829-S3DS_R1C3-200009812586_01_P001_2_0TvytxAlHg.tif
    452188     Maxar-50cm_SKYWATCH_25SEP08162829-S3DS_R1C1-200009812586_01_P001_0_omjw2boqJK.tif
    """
    
    # Parse mapping into dictionary: {filename_without_ext: numerical_id}
    filename_to_id = {}
    for line in mapping_str.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 2:
            numerical_id = parts[0]
            filename = parts[1]
            # Store both with and without extension
            filename_to_id[filename] = numerical_id
            filename_to_id[Path(filename).stem] = numerical_id
    
    # Get numerical ID for this image (fallback to filename if not in mapping)
    image_filename = image_path.name
    image_stem = image_path.stem
    image_id = filename_to_id.get(image_filename, filename_to_id.get(image_stem, image_stem))
    
    return {
        'id': 'null',  # Will be assigned by database
        'model_version': 'YOLOv11',
        'cycle_start': config.get('cycle_start', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        'cycle_end': config.get('cycle_end', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        'project_id': '85',
        'model_geometry': f'POLYGON(({ul_x} {ul_y}, {lr_x} {ul_y}, {lr_x} {lr_y}, {ul_x} {lr_y}, {ul_x} {ul_y}))',
        'model_class': building_type,
        'insert_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'detection_date': 'null',
        'reference_date': 'null',
        'detection_image': image_id,
        'reference_image_id': 'null',
        'reported_geometry': 'null',
        'reported_class': 'null',
        'reported_timestamp': 'null',
        'confirmation_status': 'null',
        'confirmed_by': 'null',
    }

def create_building_analysis_record(flag, image_path):
    # Mapping of numerical IDs to filenames
    mapping_str = """
    452121     Maxar-50cm_SKYWATCH_25SEP08162820-S3DS_R1C2-200009812612_01_P001_1_px4oFlukr3.tif
    452112     Maxar-50cm_SKYWATCH_25SEP08162820-S3DS_R1C1-200009812612_01_P001_0_BYIuWJ67y9.tif
    452125     Maxar-50cm_SKYWATCH_25SEP08162820-S3DS_R1C3-200009812612_01_P001_2_BRgxfIJ9vh.tif
    452189     Maxar-50cm_SKYWATCH_25SEP08162829-S3DS_R1C2-200009812586_01_P001_1_q4epLnOYnX.tif
    452190     Maxar-50cm_SKYWATCH_25SEP08162829-S3DS_R1C3-200009812586_01_P001_2_0TvytxAlHg.tif
    452188     Maxar-50cm_SKYWATCH_25SEP08162829-S3DS_R1C1-200009812586_01_P001_0_omjw2boqJK.tif
    """
    
    # Parse mapping into dictionary: {filename_without_ext: numerical_id}
    filename_to_id = {}
    for line in mapping_str.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 2:
            numerical_id = parts[0]
            filename = parts[1]
            # Store both with and without extension
            filename_to_id[filename] = numerical_id
            filename_to_id[Path(filename).stem] = numerical_id
    
    # Get numerical ID for this image (fallback to filename if not in mapping)
    image_filename = image_path.name
    image_stem = image_path.stem
    image_id = filename_to_id.get(image_filename, filename_to_id.get(image_stem, image_stem))
    return {
        'id': 'null',  # Will be assigned by database
        'analyzed_by': 'null',
        'optical_image_id': image_id,
        'analysis_start': 'null',
        'analysis_end': 'null',
        'status': 'null',
        'ml_status': flag,
        'project_id': '85',
    }

def process_single_image(image_path, config):
    """
    Process a single image: predictions, baseline comparison, and record extraction.
    
    Args:
        image_path (Path): Path to the image file
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (new_buildings_list, removed_buildings_list, success_flag)
    """
    print(f"\n{'='*80}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*80}")
    
    # Check if image is georeferenced
    if not is_georeferenced(image_path):
        print(f"⚠ Warning: Image is not georeferenced - skipping")
        return [], [], False
    
    print(f"✓ Image is georeferenced - proceeding with analysis")
    
    try:
        # Step 1: Generate predictions
        print("\nGenerating predictions...")
        prediction, confidence = single_image_pred(
            model_type='kfolds',
            model_version='m',
            image_id=image_path,
            sliding_window=True,
            conf_threshold=0.4,
            output_dir=config['output_dir']
        )
        
        # Step 2: Visualize predictions
        print("\nCreating visualization...")
        visualize_predictions(
            image_file=image_path,
            predictions=prediction,
            confidences=confidence,
        )
        
        # Step 3: Convert predictions to GeoJSON
        print("\nConverting predictions to GeoJSON...")
        geojson_output_path = Path(config['output_dir']) / f"{image_path.stem}_predictions.geojson"
        predictions_to_geojson(
            image_file=image_path,
            predictions=prediction,
            confidences=confidence,
            output_path=geojson_output_path
        )
        print(f"  Saved: {geojson_output_path}")
        
        # Step 4: Run baseline comparison
        print("\nRunning baseline comparison...")
        combined_geojson = baseline_comparison_geo(
            pred_geojson=geojson_output_path,
            truth_geojson=config['baseline'],
            image_path=image_path,
            output_dir=config['output_dir'],
            pipeline_shp_path=config['pipeline_path'],
            max_distance=250,
            point_distance_tolerance=10,
            save_images=False
        )

        # Step 5: Filter new buildings by confidence

        print("\nFiltering new buildings by confidence...")
        filtered_geojson, _ = filter_false_positives_by_confidence(
            geojson_path=combined_geojson,
            fp_confidence_threshold=0.6,
        )
        print(f"  Saved: {filtered_geojson}")
        # Step 5: Extract new and removed buildings for CSV export
        print("\nExtracting building records...")
        new_buildings = []
        removed_buildings = []
        
        # Load the comparison GeoJSON to extract new and removed buildings
        comparison_gdf = gpd.read_file(filtered_geojson)
        
        # Read total count from GeoJSON metadata (more efficient than extracting all features)
        with open(filtered_geojson, 'r') as f:
            geojson_data = json.load(f)
            total_count = geojson_data.get('properties', {}).get('total_features', len(comparison_gdf))
            
        
        
        
        # Get image transform for coordinate conversion
        with rasterio.open(image_path) as src:
            transform = src.transform
            
        # Extract new buildings
        fp_features = comparison_gdf[comparison_gdf['type'] == 'New']
        print(f"  Found {len(fp_features)} new buildings")

        for _, row in fp_features.iterrows():
            geom = row.geometry
            bounds = geom.bounds  # minx, miny, maxx, maxy in geographic coords
            
            # Convert back to pixel (inverse transform)
            inv_transform = ~transform
            x1, y1 = inv_transform * (bounds[0], bounds[3])  # upper left
            x2, y2 = inv_transform * (bounds[2], bounds[1])  # lower right
            
            record = create_building_record(
                [x1, y1, x2, y2], transform, image_path, 'New', config
            )
            new_buildings.append(record)

        # Extract removed buildings
        fn_features = comparison_gdf[comparison_gdf['type'] == 'Removed']
        print(f"  Found {len(fn_features)} removed buildings")
        
        for _, row in fn_features.iterrows():
            geom = row.geometry
            bounds = geom.bounds
            
            inv_transform = ~transform
            x1, y1 = inv_transform * (bounds[0], bounds[3])
            x2, y2 = inv_transform * (bounds[2], bounds[1])
            
            record = create_building_record(
                [x1, y1, x2, y2], transform, image_path, 'Removed', config
            )
            removed_buildings.append(record)
        
        print(f"\n✓ Successfully processed {image_path.name}")
        
        return new_buildings, removed_buildings, total_count, True
        
    except Exception as e:
        print(f"\n✗ Error processing {image_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return [], [], [], False

def main():
    """
    Main processing function: processes all images and exports combined CSV.
    """
    print("\n" + "="*80)
    print("BATCH IMAGE PROCESSING & BASELINE COMPARISON")
    print("="*80)
    
    # Get list of images to process
    image_list = get_image_list(CONFIG['input_images'])
    
    if not image_list:
        print("No images found to process!")
        return
    
    print(f"\nFound {len(image_list)} image(s) to process")
    for img in image_list:
        print(f"  - {img.name}")
    
    # Process each image
    all_new_buildings = []
    all_removed_buildings = []
    successful_images = []
    failed_images = []
    
    for image_path in image_list:
        output_path = Path(CONFIG['output_dir']) / f'sw_predictions_1024_0.5/'
        if output_path.exists():
            shutil.rmtree(output_path)
            print(f"Deleted existing prediction folder: {output_path}")

        new_buildings, removed_buildings, total_count, success = process_single_image(image_path, CONFIG)

        if success:
            all_new_buildings.extend(new_buildings)
            all_removed_buildings.extend(removed_buildings)
            successful_images.append(image_path.name)
        else:
            failed_images.append(image_path.name)
        flag = flagging(new_buildings, removed_buildings, total_count, criteria=0.3)
        analysis_record = create_building_analysis_record(flag, image_path)
        analysis_df = pd.DataFrame([analysis_record])
        analysis_csv_path = Path(CONFIG['output_dir']) / "building_analysis.csv"  # Fixed filename

        # Write header only if file doesn't exist
        analysis_df.to_csv(
            analysis_csv_path, 
            mode='a',  # Always append
            header=not analysis_csv_path.exists(),  # Header only if new file
            index=False
        )

    
    # Export combined CSV
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    
    combined_buildings = all_new_buildings + all_removed_buildings
    
    if combined_buildings:
        df = pd.DataFrame(combined_buildings)
        csv_path = Path(CONFIG['output_dir']) / f"building_changes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Exported combined CSV: {csv_path}")
        print(f"  Total records: {len(combined_buildings)}")
        print(f"    - New buildings: {len(all_new_buildings)}")
        print(f"    - Removed buildings: {len(all_removed_buildings)}")
    else:
        print("\n⚠ No building changes detected across all images")
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total images: {len(image_list)}")
    print(f"Successful: {len(successful_images)}")
    print(f"Failed: {len(failed_images)}")
    
    if failed_images:
        print("\nFailed images:")
        for img in failed_images:
            print(f"  - {img}")
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

