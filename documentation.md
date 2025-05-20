# Introduction

This is documentation to run the object detection algorithm. There are some parts that are generalized and some parts that are specific and depending on the part it will be marked as to which category it falls. The basis is to run different object detections algorithms using the RCNN model and the YOLO model. Further analysis is done with the results of the YOLO model, as this was determined to be better than the former. This document will run through the entire process from model training until testing results and analysis.

# Using the YOLO model
## 1. Dataset preperation

If using the YOLO model, the dataset needs to be prepared in a specific way. The structure of the files should be in this format for the images and labels folders:
```
parent/
│
├──images/
│  ├─ image1.jpg
│  └─ image2.jpg     
├──labels/
│   ├─ image1.txt
│   └─ image2.txt
├─autosplit_train.txt
├─autosplit_val.txt
├─autosplit_test.txt
└─data.yaml

```
First, there needs to be a split between the train, validation and test set. This can be done easily by calling the autosplit function:
```python
from ultralytics.data.utils import autosplit

# assuming a path to images
images = Path('path/to/images/')
autosplit(images, weights=(0.7,0.15,0.15)) # train, validation, test splits

```
This creates autosplit.txt files containing the names of label files for each split in the parent folder of the labels. This information should then be used to create a YAML file that contains the basic information for running the training. The YAML contains the path (or relative path) to the parent, information about the splits, and the class labels for the dataset as shown in this example:

```yaml

path: .
train: autosplit_train.txt
val: autosplit_val.txt
names: 
    0: Building
```
The YOLO labelling format maps a number to the class name using this YAML file. The labelling format itself for horizontal bounding boxes contains 5 numbers: class_number upper_left_x_normalized_coordinate, upper_left_y_normalized_coordinate, lower_right_x_normalized_coordinate, lower_right_y_normalized_coordinate. For example:
```
0 0.1 0.5 0.15 0.53
2 0.3 0.3 0.5 0.5
```

For more information check the YOLO documentation from Ultralytics
Once the data is formatted into this way it is ready to be used for training.

## 2. Training

To train the data with the YOLO model is simple. The information can be found on the YOLO [documentation](https://docs.ultralytics.com/modes/train/) but to summarize there is just a couple lines of code that need to be run:
```python
from ultralytics import YOLO # or any other model in their codebase

model = YOLO('yolo11n.pt') 
results = model.train()
```
There are various parameters that can be modified within the train function that are detailed in the documentations. Hyperparameters can also be modified within this function, or by changing the defaults found in ultralytics/ultralytics/cfg/default.yaml. When running the training a destination folder can be specified, or it will create a default folder which will contain the weights of the best and last epoch, and some more analytics that can be used to gain insight on the training process. This function also does validation but using model.val() can also be used.

## 3. Predictions

Testing the YOLO model is also very simple. The basic command can be summarized as follows:

```python

from ultralytics import YOLO
model = YOLO('path/to/best/weights.pt')
results = model.predict()
```

Further usage of the predict method can be seen on the [documentation](https://docs.ultralytics.com/modes/predict/).

And with these steps, that is most of the functionalities of the YOLO model from ultralytics.

# Usage of my model

There are various steps that are required to be able to use the workflow of this model. The first step was to use pretrain the YOLOv11m model on the xView dataset. Then using the best weights from this, the model is retrained on customer data using k-means cross validation. Using the k_means.py the paths to the customer images and the path to the best weights from the pretained model are input and script should be able to handle the training and will output the paths and the training analytics to a folder based on the base model YOLO model (n, m, etc...).

Then to do the post processing analytics the BHE_postprocessing.ipynb notebook is used. This notebook is designed so that any changes should be made in the first configuration cell and the rest of the cells can just be run as is. In its current state it is to be used for the BHE dataset with provided shapefiles to give truth building outlines.

# Usage of RCNN model

The RCNN model is a bit different in its usage. The usage guidelines are already available, and following those the same steps can be taken (pretraining on the xView then retraining on the customer dataset). If the RCNN is used then the predictions are already made by the RCNN model and so only the predictions themselves need to be loaded into the notebook for the post processing. The reason for this is that the format of the RCNN prediction outputs are very different since it is creating rotated bounding boxes and so the script to run the predictions for the RCNN handles this and converts the predictions to the YOLO format so that it can be accurately compared. 


# Post processing notes

The post-processing logic currently assumes that there will always be a baseline to predict against, in the form of a shapefile. In the current version, the shapefile can either contain polygons or points and the script is able to handle both scenarios. After the initialization of the variables that can be adjusted, the predictions are made using the commands from the YOLO documentation and these are stored in a file. The predictions can either be made directly on the images or by a sliding window. The meat of the program is the analysis of the predictions. First, information relating to the pipeline shapefile is loaded and a buffer is created to define the corridor width. Then each prediction is compared to the baseline, either the point or the polygon. In the event of a polygon, there is likely a point associated with it as well and if this is the case then the prediction is only compared to the polygon so as to prevent a double negative. Furthermore, predictions over the same area are combined and only the prediction that has its centre closest to the centre of the baseline polygon or the point is take **subject to change for better logic**. Then for the remaining predictions that exist after all the baseline objects have been matched, they are considered as false positives. Because this is always being compared to the baseline, there is a filter that can be adjusted to only allow false positives above a certain confidence level to pass through, so as to remove noise. The false positives can also be considered as newly found buildings, however given the current state of the model, the vast majority are actual false positives. Similarly, any objects in the baseline that were not found by the model can be considered as removed buildings, although, again given the state of the model, most of these do exist, but were not found by the model. Many metrics for each of these are gathered as well, and they are all processed and given in the form of text files or csvs which can be used to keep track of the post processing data, especially, where the model does not do well. As part of the processing, there is a flag that can be triggered if there are more than 10 buildings in an image and either 30% of the buildings are missed, or they are false positives. In this case it will be flagged and the image id will be output to a text file which can be used for manual checking of the entire image. These scenarios occur, if the image is not good quality, there is a co-registration issue, clouds etc... This can be improved in the future, but this is just a barebones step as the workflow gets determined. The output of the model can either be an image that shows the bounding boxes for each image prediction, or to save resources, it will just be a text file that shows the image bounding box coordinates for each of the images but does not output an image. This is usually used for a sensitivity analysis where multiple runs are needed, and thus multiple copies of each image will be generated otherwise.


