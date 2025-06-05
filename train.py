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



autosplit_train = Path('/cephfs/work/rithvik/datasets/datasets/NatCombined/images/autosplit_train.txt')
autosplit_val = Path('/cephfs/work/rithvik/datasets/datasets/NatCombined/images/autosplit_val.txt')
if not autosplit_train.exists() or not autosplit_val.exists():
    images = Path('/cephfs/work/rithvik/datasets/datasets/NatCombined/images')

    # Split
    autosplit(images, weights=(0.7, 0.15, 0.15))
else:
    print("autosplit_train.txt and autosplit_val.txt already exist. Skipping conversion and splitting.")

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

model = YOLO('yolo11m.pt')   # switch to whichever YOLO model from worst to best (n, s, m, l, x) 
# hyp = {

# }

results = model.train(data='./YOLO/combined.yaml', 
                      epochs=100, batch=1, imgsz=1024, 
                      workers=16, scale=0.1, project='./pre_trained/', 
                      name='xView_combined', resume=False) # This does both training and validation but model.val can also be used to validate the model


# model = RTDETR('/home/rithvik/YOLO/test_runs/detect/xView_combined_DETR4/weights/last.pt')
# args = dict(model='/home/rithvik/YOLO/test_runs/detect/xView_combined/weights/best.pt', data='/cephfs/work/rithvik/datasets/datasets/NatFuel_NatGrid_buildings_dataset/NatFuel_Datasplit/trainval_YOLO/NatFuel.yaml', 
#                       epochs=200, batch=4, imgsz=1024, workers=16, scale=0.1,
#                       project='/home/rithvik/YOLO/test_runs/detect/', name='xView_combined_Natfuel', 
#                       )
# trainer = CustomDetectionTrainer(overrides=args) (doesn't work)
# results = trainer.train()