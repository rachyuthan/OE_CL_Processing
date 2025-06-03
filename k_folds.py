from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold
import datetime
import shutil
from ultralytics import YOLO
from PIL import Image


dataset_path = Path("/cephfs/work/rithvik/datasets/datasets/NatCombined/cropped/")
supported_extensions = [".jpg", ".jpeg", ".png"]

# Initialize an empty list to store image file paths
images = []
image_dir = dataset_path / "images" # Assuming images are in images/
label_dir = dataset_path / "labels" # Assuming labels are in labels/

# Loop through supported extensions and gather image files
for ext in supported_extensions:
    images.extend(sorted(image_dir.rglob(f"*{ext}")))

# Ensure corresponding label files exist, create empty ones if not
for img_path in images:
    label_path = label_dir / (img_path.stem + ".txt")
    if not label_path.exists():
        print(f"Label file missing for {img_path.name}, creating empty file: {label_path}")
        label_path.touch() # Create an empty file

# Now gather the labels, including any newly created empty ones
labels = sorted(label_dir.glob("*.txt"))

# Ensure the number of images and labels match after potential creation
if len(images) != len(labels):
    print(f"Warning: Mismatch after label creation check. Images: {len(images)}, Labels: {len(labels)}")
    labels = [label_dir / (img.stem + ".txt") for img in images]


yaml_file = "/cephfs/work/rithvik/datasets/datasets/NatCombined/data.yaml"
with open(yaml_file, "r", encoding="utf-8") as f:
    classes = yaml.safe_load(f)["names"]

cls_idx = sorted(classes.keys())

index = [label.stem for label in labels]
labels_df = pd.DataFrame([], columns=cls_idx, index=index)

for label in labels:
    lbl_counter = Counter()

    with open(label,"r") as lf:
        lines = lf.readlines()
    
    for line in lines:
        # Skip empty or whitespace-only lines
        line = line.strip() 
        if not line:
            continue
        
        # Process valid lines
        try: # Add try-except for robustness against malformed lines
            lbl_counter[int(line.split(" ")[0])] += 1
        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping malformed line in {label.name}: '{line}'. Error: {e}")

    labels_df.loc[label.stem] = lbl_counter

labels_df = labels_df.fillna(0.0)

# Calculate and print total label counts
print("Total label counts across the dataset:")
total_counts = labels_df.sum()
for class_id in cls_idx:
    class_name = classes[class_id]
    count = int(total_counts[class_id])
    print(f"- {class_name} ({class_id}): {count}")
print("-" * 30) # Separator

ksplit = 5

kf = KFold(n_splits=ksplit, shuffle=True, random_state=42)

kfolds = list(kf.split(labels_df))

folds = [f"split_{n}" for n in range(1, ksplit +1)]
folds_df = pd.DataFrame(index=index, columns=folds)

for i, (train,val) in enumerate(kfolds, start=1):
    folds_df[f"split_{i}"].loc[labels_df.iloc[train].index] = "train"
    folds_df[f"split_{i}"].loc[labels_df.iloc[val].index] = "val"

fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1e-7)
    fold_lbl_distrb.loc[f"split_{n}"] = ratio

# Create the necessary directories and dataset YAML files (unchanged)
save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
save_path.mkdir(parents=True, exist_ok=True)
ds_yamls = []

for split in folds_df.columns:
    # Create directories
    split_dir = save_path / split
    if not split_dir.exists():
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # Create dataset YAML files
    dataset_yaml = split_dir / f"{split}_dataset.yaml"
    ds_yamls.append(dataset_yaml)

    with open(dataset_yaml, "w") as ds_y:
        yaml.safe_dump(
            {
                "path": split_dir.as_posix(),
                "train": "train",
                "val": "val",
                "names": classes,
            },
            ds_y,
        )

for image, label in zip(images, labels):
    # Check if the label file actually exists before proceeding (robustness)
    if not label.exists():
        print(f"Skipping copy for {image.name} as label {label.name} was expected but not found.")
        continue # Skip this pair if the label file is unexpectedly missing

    # Ensure stems match before copying (extra safety check)
    if image.stem != label.stem:
        print(f"Warning: Stem mismatch detected! Image: {image.stem}, Label: {label.stem}. Skipping this pair.")
        continue

    for split, k_split in folds_df.loc[image.stem].items():
        # Destination directory
        img_to_path = save_path / split / k_split / "images"
        lbl_to_path = save_path / split / k_split / "labels"

        # Create directories if they don't exist (redundant if created before, but safe)
        img_to_path.mkdir(parents=True, exist_ok=True)
        lbl_to_path.mkdir(parents=True, exist_ok=True)

        # Destination files
        dest_img = img_to_path / image.name
        dest_lbl = lbl_to_path / label.name

        # Copy files
        if not dest_img.exists():
            shutil.copy(image, dest_img)
        # Copy label file even if it's empty (created earlier)
        if not dest_lbl.exists():
            shutil.copy(label, dest_lbl)

print("Dataset split and copied successfully")

path = "./OE_CL_Processing/pre_trained/weights/best.pt" #path to pre trained YOLO model
model = YOLO(path, task='detect')
results = {}

batch = 8
project = "YOLO/k_folds_cross_val_m"
epochs = 200

for k in range(ksplit):
    dataset_yaml = ds_yamls[k]
    model = YOLO(path, task='detect')
    model.train(data=dataset_yaml, epochs=epochs, batch=batch, imgsz=1024, project=project, name=f"split_{k+1}")
    results[k] = model.metrics



    def crop_image(image_path, crop_size, overlap):
        """
        Crop an input image into smaller square images with a specified crop size and overlap.

        Args:
            image_path (str or Path): Path to the input image.
            crop_size (int): Size of the square crop (e.g., 256 for 256x256 crops).
            overlap (int): Overlap between crops in pixels.

        Returns:
            List of cropped images as PIL.Image objects.
        """

        image = Image.open(image_path)
        width, height = image.size
        crops = []

        step = crop_size - overlap
        for top in range(0, height, step):
            for left in range(0, width, step):
                # Ensure the crop doesn't exceed the image boundaries
                right = min(left + crop_size, width)
                bottom = min(top + crop_size, height)
                crop = image.crop((left, top, right, bottom))
                crops.append(crop)

        return crops