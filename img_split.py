import cv2
from pathlib import Path
from tqdm import tqdm
import json
from shapely.geometry import box, shape
import rasterio
from rasterio.transform import rowcol

def parse_yolo_label(label_line):
    """Parse a YOLO format label line"""
    parts = label_line.strip().split()
    class_id = int(parts[0])
    x_center, y_center, width, height = map(float, parts[1:5])
    return class_id, x_center, y_center, width, height

def yolo_to_absolute(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO normalized coordinates to absolute coordinates"""
    x1 = (x_center - width/2) * img_width
    y1 = (y_center - height/2) * img_height
    x2 = (x_center + width/2) * img_width
    y2 = (y_center + height/2) * img_height
    return x1, y1, x2, y2

def absolute_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """Convert absolute coordinates to YOLO normalized coordinates"""
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

def crop_and_adjust_labels(image_path, label_path, crop_size, padding, images_output_dir, labels_output_dir, valid_areas_geom=None, image_transform=None):
    """Crop image into squares and adjust labels accordingly"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return 0, 0
    
    img_height, img_width = image.shape[:2]
    
    # Read labels
    labels = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    labels.append(parse_yolo_label(line))
    
    # Calculate crop positions with padding
    step_size = crop_size - 2 * padding
    crop_count = 0
    skipped_count = 0
    
    for y in range(0, img_height, step_size):
        for x in range(0, img_width, step_size):
            # Calculate crop boundaries
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_width, x1 + crop_size)
            y2 = min(img_height, y1 + crop_size)
            
            # Adjust if crop would be smaller than desired size
            if x2 - x1 < crop_size:
                x1 = max(0, x2 - crop_size)
            if y2 - y1 < crop_size:
                y1 = max(0, y2 - crop_size)
            
            # Check if crop intersects with valid areas
            if valid_areas_geom is not None and image_transform is not None:
                # Convert pixel coordinates to geographic coordinates
                # Top-left corner
                geo_x1, geo_y1 = image_transform * (x1, y1)
                # Bottom-right corner
                geo_x2, geo_y2 = image_transform * (x2, y2)
                
                # Create crop box in geographic coordinates
                crop_box = box(geo_x1, geo_y2, geo_x2, geo_y1)
                
                # Check intersection with valid areas
                if not valid_areas_geom.intersects(crop_box):
                    skipped_count += 1
                    continue
            
            # Crop image
            crop = image[y1:y2, x1:x2]
            crop_height, crop_width = crop.shape[:2]
            
            # Process labels for this crop
            crop_labels = []
            for class_id, x_center, y_center, width, height in labels:
                # Convert to absolute coordinates
                abs_x1, abs_y1, abs_x2, abs_y2 = yolo_to_absolute(
                    x_center, y_center, width, height, img_width, img_height
                )
                
                # Check if label intersects with crop
                if (abs_x2 > x1 and abs_x1 < x2 and abs_y2 > y1 and abs_y1 < y2):
                    # Clip coordinates to crop boundaries
                    clipped_x1 = max(abs_x1, x1) - x1
                    clipped_y1 = max(abs_y1, y1) - y1
                    clipped_x2 = min(abs_x2, x2) - x1
                    clipped_y2 = min(abs_y2, y2) - y1
                    
                    # Convert back to YOLO format relative to crop
                    norm_x_center, norm_y_center, norm_width, norm_height = absolute_to_yolo(
                        clipped_x1, clipped_y1, clipped_x2, clipped_y2, crop_width, crop_height
                    )
                    
                    # Only keep if the clipped box has reasonable size
                    if norm_width > 0.01 and norm_height > 0.01:
                        crop_labels.append((class_id, norm_x_center, norm_y_center, norm_width, norm_height))
            
            # Save crop and labels
            if crop_labels or True:  # Save even if no labels
                stem = image_path.stem
                crop_image_path = images_output_dir / f"{stem}_crop_{crop_count}.png"
                crop_label_path = labels_output_dir / f"{stem}_crop_{crop_count}.txt"
                
                cv2.imwrite(str(crop_image_path), crop)
                
                with open(crop_label_path, 'w') as f:
                    for label in crop_labels:
                        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
                
                crop_count += 1
    
    return crop_count, skipped_count

def process_dataset(image_dir, label_dir, output_dir, crop_size, padding, valid_areas_dir=None):
    """Process entire dataset"""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    
    # Create separate directories for images and labels
    images_output_dir = output_dir / "images"
    labels_output_dir = output_dir / "labels"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)
    
    if valid_areas_dir:
        valid_areas_dir = Path(valid_areas_dir)
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]

    total_crops = 0
    total_skipped = 0

    for image_path in tqdm(image_files, desc="Processing images"):
        # Find corresponding label file
        label_path = label_dir / f"{image_path.stem}.txt"
        
        # Load valid areas if provided
        valid_areas_geom = None
        image_transform = None
        
        if valid_areas_dir:
            valid_areas_path = valid_areas_dir / f"{image_path.stem}.geojson"
            
            if valid_areas_path.exists():
                try:
                    with open(valid_areas_path, 'r') as f:
                        valid_areas_geojson = json.load(f)
                    
                    if valid_areas_geojson['type'] == 'FeatureCollection':
                        valid_areas_geom = shape(valid_areas_geojson['features'][0]['geometry'])
                    else:
                        valid_areas_geom = shape(valid_areas_geojson['geometry'])
                    
                    # Get image transform for coordinate conversion
                    with rasterio.open(str(image_path)) as src:
                        image_transform = src.transform
                        
                except Exception as e:
                    print(f"Warning: Could not load valid areas for {image_path.stem}: {e}")
        
        crop_count, skipped_count = crop_and_adjust_labels(
            image_path, label_path, crop_size, padding, 
            images_output_dir, labels_output_dir,
            valid_areas_geom, image_transform
        )
        total_crops += crop_count
        total_skipped += skipped_count
        
    print(f"Processed {len(image_files)} images.")
    print(f"Created {total_crops} crops.")
    if valid_areas_dir:
        print(f"Skipped {total_skipped} crops (outside valid areas).")

def main():
    
    CONFIG = {
        "image_dir": "/cephfs/work/rithvik/datasets/datasets/BHE/Maxar_GBA_subset/images/",
        "label_dir": "/cephfs/work/rithvik/datasets/datasets/BHE/Maxar_GBA_subset/labels/",
        "output_dir": "/cephfs/work/rithvik/datasets/datasets/BHE/Maxar_images_cropped/",
        "valid_areas_dir": "/cephfs/work/rithvik/datasets/datasets/BHE/Maxar_images_filtered/valid_areas",
        "crop_size": 1024,
        "padding": 50,
        
    }


    process_dataset(
        CONFIG["image_dir"], 
        CONFIG["label_dir"], 
        CONFIG["output_dir"], 
        CONFIG["crop_size"], 
        CONFIG["padding"],
        CONFIG.get("valid_areas_dir")
    )
    print("Processing complete!")

if __name__ == "__main__":
    main()