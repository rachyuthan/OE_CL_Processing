from ultralytics import YOLO, RTDETR

import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import yaml

from ultralytics.data.utils import autosplit
from ultralytics.utils.ops import xyxy2xywhn

def convert_labels():
    # Convert xView geoJSON labels to YOLO format
    path = Path('/home/rithvik/datasets/xView_full')

    # Make dirs
    labels = Path(path / 'labels' / 'train')
    os.system(f'rm -rf {labels}')
    labels.mkdir(parents=True, exist_ok=True)

    # xView classes 11-94 to 0-59
    xview_class2index = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10, 11,
                         12, 13, 14, 15, -1, -1, 16, 17, 18, 19, 20, 21, 22, -1, 23, 24, 25, -1, 26, 27, -1, 28, -1,
                         29, 30, 31, 32, 33, 34, 35, 36, 37, -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1, -1, -1, 46,
                         47, 48, 49, -1, 50, 51, -1, 52, -1, -1, -1, 53, 54, -1, 55, -1, -1, 56, -1, 57, -1, 58, 59]

    shapes = {}
    for feature in tqdm(data['features'], desc=f'Converting labels'):
        p = feature['properties']
        if p['bounds_imcoords']:
            id = p['image_id']
            file = path / 'train_images' / id
            if file.exists():  # 1395.tif missing
                try:
                    coords = p['bounds_imcoords']
                    class_number = int(p['type_id'])
                    if class_number in xview_class2index and xview_class2index[class_number] != -1:
                        yolo_class = xview_class2index[class_number]
                        data = f"{coords} {yolo_class} 0"
                        yolo_format = convert_to_yolo_format(data, 1024, 1024)

                        # Write YOLO label
                        if id not in shapes:
                            shapes[id] = Image.open(file).size
                        with open((labels / id).with_suffix('.txt'), 'a') as f:
                            f.write(yolo_format + '\n')
                except Exception as e:
                    print(f'WARNING: skipping one label for {file}: {e}')
                    

def convert_to_yolo_format(data, img_width, img_height):
    # Extract the coordinates
    x1, y1, _, _, x3, y3, _, _, class_number, _ = data.split()
    x1, y1, x3, y3 = float(x1), float(y1), float(x3), float(y3)

    # Convert to (x1, y1, x2, y2) format
    x2, y2 = x3, y3
    box = np.array([[x1, y1, x2, y2]])

    xview_class2index = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10, 11,
                         12, 13, 14, 15, -1, -1, 16, 17, 18, 19, 20, 21, 22, -1, 23, 24, 25, -1, 26, 27, -1, 28, -1,
                         29, 30, 31, 32, 33, 34, 35, 36, 37, -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1, -1, -1, 46,
                         47, 48, 49, -1, 50, 51, -1, 52, -1, -1, -1, 53, 54, -1, 55, -1, -1, 56, -1, 57, -1, 58, 59]

    # Convert xView class number to YOLO class number
    class_number = int(class_number)
    yolo_class_number = xview_class2index[class_number]

    # Convert to YOLO format using xyxy2xywhn
    yolo_box = xyxy2xywhn(box, w=img_width, h=img_height)

    # Format the output
    yolo_format = f"{yolo_class_number} {yolo_box[0, 0]:.6f} {yolo_box[0, 1]:.6f} {yolo_box[0, 2]:.6f} {yolo_box[0, 3]:.6f}"
    return yolo_format

def process_file(file_path, img_width, img_height):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            yolo_format = convert_to_yolo_format(line.strip(), img_width, img_height)
            file.write(yolo_format + '\n')



# ============================================================================
# CONFIGURATION - Update these paths for your dataset
# ============================================================================
CONFIG = {
    "dataset_name": "Maxar_images-skysat_combined",  # Name for your dataset
    # Option 1: Single directory (original behavior)
    # "images_dir": "/cephfs/work/rithvik/datasets/datasets/BHE/Maxar_images_cropped/images/",
    # "labels_dir": "/cephfs/work/rithvik/datasets/datasets/BHE/Maxar_images_cropped/labels/",
    
    # Option 2: Multiple directories (comment out images_dir/labels_dir above and use this)
    "datasets": [
        {
            "images": "/cephfs/work/rithvik/datasets/datasets/BHE/Maxar_images_cropped/images/",
            "labels": "/cephfs/work/rithvik/datasets/datasets/BHE/Maxar_images_cropped/labels/",
        },
        {
            "images": "/cephfs/work/rithvik/datasets/datasets/Sept2025Dataset/images/",
            "labels": "/cephfs/work/rithvik/datasets/datasets/Sept2025Dataset/labels/",
        },
    ],
    
    "dataset_root": "/cephfs/work/rithvik/datasets/datasets/BHE/Maxar_images_skysat_combined",  # Where YAML will be saved
    "split_weights": (0.85, 0.15, 0.00),  # train, val, test split ratios
    "num_classes": 1,  # Number of classes in your dataset
    "class_names": ["building"],  # List of class names
}
# ============================================================================

def create_dataset_yaml(config):
    """Create YAML file for YOLO training"""
    dataset_root = Path(config["dataset_root"])
    yaml_path = dataset_root / "dataset.yaml"
    
    # Create names dictionary with index mapping (e.g., {0: 'building', 1: 'car'})
    names_dict = {i: name for i, name in enumerate(config["class_names"])}
    
    # Create YAML content
    yaml_content = {
        "path": str(dataset_root.absolute()),
        "train": "autosplit_train.txt",
        "val": "autosplit_val.txt",
    }
    
    # Add test split if it exists
    if config["split_weights"][2] > 0:
        yaml_content["test"] = "autosplit_test.txt"
    
    # Add names mapping
    yaml_content["names"] = names_dict
    
    # Write YAML file with custom formatting to match YOLO format
    with open(yaml_path, 'w') as f:
        f.write(f"path: {yaml_content['path']}\n")
        f.write(f"train: {yaml_content['train']}\n")
        f.write(f"val: {yaml_content['val']}\n")
        if "test" in yaml_content:
            f.write(f"test: {yaml_content['test']}\n")
        f.write("names:\n")
        for idx, name in names_dict.items():
            f.write(f"  {idx}: {name}\n")
    
    print(f"Created dataset YAML file: {yaml_path}")
    return yaml_path

def collect_image_paths(config):
    """Collect all image paths from single or multiple dataset directories"""
    image_paths = []
    
    # Check if using single directory (old format) or multiple directories
    if "images_dir" in config:
        # Single directory mode
        images_dir = Path(config["images_dir"])
        labels_dir = Path(config["labels_dir"])
        
        if not images_dir.exists():
            print(f"ERROR: Images directory does not exist: {images_dir}")
            exit(1)
        if not labels_dir.exists():
            print(f"ERROR: Labels directory does not exist: {labels_dir}")
            exit(1)
        
        # Collect all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            image_paths.extend(images_dir.glob(ext))
        
        print(f"Found {len(image_paths)} images in: {images_dir}")
        
    elif "datasets" in config:
        # Multiple directories mode
        for i, dataset in enumerate(config["datasets"], 1):
            images_dir = Path(dataset["images"])
            labels_dir = Path(dataset["labels"])
            
            if not images_dir.exists():
                print(f"ERROR: Images directory {i} does not exist: {images_dir}")
                exit(1)
            if not labels_dir.exists():
                print(f"ERROR: Labels directory {i} does not exist: {labels_dir}")
                exit(1)
            
            # Collect all image files from this dataset
            dataset_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                dataset_images.extend(images_dir.glob(ext))
            
            image_paths.extend(dataset_images)
            print(f"Found {len(dataset_images)} images in dataset {i}: {images_dir}")
        
        print(f"Total images across all datasets: {len(image_paths)}")
    else:
        print("ERROR: Config must contain either 'images_dir' or 'datasets'")
        exit(1)
    
    return image_paths

def create_custom_autosplit(image_paths, dataset_root, weights=(0.8, 0.1, 0.1)):
    """Create autosplit files from a list of image paths"""
    import random
    
    # Shuffle images
    random.seed(42)  # For reproducibility
    image_paths = list(image_paths)
    random.shuffle(image_paths)
    
    # Calculate split indices
    n = len(image_paths)
    train_split = int(n * weights[0])
    val_split = int(n * (weights[0] + weights[1]))
    
    # Split the data
    train_images = image_paths[:train_split]
    val_images = image_paths[train_split:val_split]
    test_images = image_paths[val_split:]
    
    # Write autosplit files
    autosplit_train = dataset_root / "autosplit_train.txt"
    autosplit_val = dataset_root / "autosplit_val.txt"
    autosplit_test = dataset_root / "autosplit_test.txt"
    
    with open(autosplit_train, 'w') as f:
        for img in train_images:
            f.write(f"{img.absolute()}\n")
    
    with open(autosplit_val, 'w') as f:
        for img in val_images:
            f.write(f"{img.absolute()}\n")
    
    if len(test_images) > 0:
        with open(autosplit_test, 'w') as f:
            for img in test_images:
                f.write(f"{img.absolute()}\n")
    
    print(f"✓ Created autosplit files:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")

# Create dataset root directory if it doesn't exist
dataset_root = Path(CONFIG["dataset_root"])
dataset_root.mkdir(parents=True, exist_ok=True)

# Define autosplit file paths
autosplit_train = dataset_root / "autosplit_train.txt"
autosplit_val = dataset_root / "autosplit_val.txt"
autosplit_test = dataset_root / "autosplit_test.txt"

# Perform autosplit if needed
if not autosplit_train.exists() or not autosplit_val.exists():
    print("Creating train/val split...")
    
    # Collect all image paths
    image_paths = collect_image_paths(CONFIG)
    
    if len(image_paths) == 0:
        print("ERROR: No images found!")
        exit(1)
    
    # Create custom autosplit
    create_custom_autosplit(image_paths, dataset_root, weights=CONFIG["split_weights"])
    print("✓ Train/val split completed")
else:
    print("✓ Autosplit files already exist. Skipping splitting.")

# Create YAML file
yaml_path = create_dataset_yaml(CONFIG)

print("\n" + "="*70)
print("DATASET CONFIGURATION")
print("="*70)
print(f"Dataset name: {CONFIG['dataset_name']}")
if "images_dir" in CONFIG:
    print(f"Images: {CONFIG['images_dir']}")
    print(f"Labels: {CONFIG['labels_dir']}")
elif "datasets" in CONFIG:
    print(f"Multiple datasets ({len(CONFIG['datasets'])}):")
    for i, ds in enumerate(CONFIG['datasets'], 1):
        print(f"  Dataset {i}:")
        print(f"    Images: {ds['images']}")
        print(f"    Labels: {ds['labels']}")
print(f"Dataset root: {dataset_root}")
print(f"YAML file: {yaml_path}")
print(f"Number of classes: {CONFIG['num_classes']}")
print(f"Class names: {CONFIG['class_names']}")
print("="*70)

print("\nContinue with training? (y/n):")
cont = input().strip().lower()
if cont == 'n':
    print("Exiting.")
    exit(0)
else:
    print("Continuing to training...")

def get_dataset_path(yaml_path):
    """Extract datset paths from config file"""

    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    base_path = Path(config['path'])
    train_path = base_path / config['train']
    val_path = base_path / config['val']
    

    return train_path, val_path

def calculate_dataset_mean_std(yaml_path, train=True):
    """Calculate mean and std of dataset"""

    train_dir, val_dir = get_dataset_path(yaml_path)
    images = []

    
    base_dir = Path(yaml_path).parent
    image_dir = train_dir if train else val_dir

    if str(image_dir).endswith('.txt'):
        # Accumulators
        sum_pixels = np.zeros(3, dtype=np.float64)
        sum_squared_diff = np.zeros(3, dtype=np.float64)
        num_pixels = 0
        with open(image_dir, 'r') as file:
            image_paths = file.readlines()
            image_paths = [Path(base_dir) / path.strip() for path in image_paths]
        for path in tqdm(image_paths, desc="Loading images"):
                image = Image.open(path)
                image_np = np.array(image)
                h, w, _ = image_np.shape

                num_pixels += h * w
                sum_pixels += np.sum(image_np, axis=(0, 1))
        mean = sum_pixels / num_pixels

        for path in tqdm(image_paths, desc="Loading images"):
            image = Image.open(path)
            image_np = np.array(image)
            
            squared_diff = (image_np - mean) ** 2
            sum_squared_diff += np.sum(squared_diff, axis=(0, 1))
        std = np.sqrt(sum_squared_diff / num_pixels)
    return mean/255, std/255

# Train the model

model = YOLO('pre_trained/weights/best.pt')   # switch to whichever YOLO model from worst to best (n, s, m, l, x) 

results = model.train(data=str(yaml_path), 
                      epochs=200, batch=4, imgsz=1024, 
                      workers=16, scale=0.1, project='./Maxar_skysat_combined/', 
                      name=CONFIG['dataset_name'], resume=False) # This does both training and validation but model.val can also be used to validate the model


# model = RTDETR('/home/rithvik/YOLO/test_runs/detect/xView_combined_DETR4/weights/last.pt')
# args = dict(model='/home/rithvik/YOLO/test_runs/detect/xView_combined/weights/best.pt', data='/cephfs/work/rithvik/datasets/datasets/NatFuel_NatGrid_buildings_dataset/NatFuel_Datasplit/trainval_YOLO/NatFuel.yaml', 
#                       epochs=200, batch=4, imgsz=1024, workers=16, scale=0.1,
#                       project='/home/rithvik/YOLO/test_runs/detect/', name='xView_combined_Natfuel', 
#                       )
# trainer = CustomDetectionTrainer(overrides=args) (doesn't work)
# results = trainer.train()