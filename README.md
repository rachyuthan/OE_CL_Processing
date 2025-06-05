# OE_CL_Processing
Scripts to run class location and postprocessing 

Run this to setup the environment.
```bash
conda env create -f environment.yaml
conda activate OE_YOLO
```

For the cosmic eye client it is required to be added seperately:
```bash
pip install --extra-index-url https://pypi.orbitaleye.nl/simple cosmic_eye_client
```

Run [Labels.py](Labels.py) to generate truth labels and pull images from server (only works for BHE). 
Run [main.py](main.py) to perform inference on the images along with postprocessing on the images. Change directories in the config so that everything is pulled from the correct locations. 
Change the CONFIG settings to play around with the confidence threshold for accepting predictions and filtering false positives. 
Run the program.

Training of the pretrained xView and Openbuildings dataset can be done with [train.py](train.py). This performs basic YOLO training with the default aumgmentation settings. Custom data augmentation is done an extra configuration yaml file. An extra parameter must be added and supported hyperparameters are given in the [YOLO docs](https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters).
```yaml
# Add this for hyperparameter changes
# Hyperparameters ------------------------------------------------------------------------------------------------------
lr0: 0.01 # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
lrf: 0.01 # (float) final learning rate (lr0 * lrf)
# ... Any other changes
```

Then when training the script needs to be called:
```python
results = model.train(cfg=config.yaml) # assuming config.yaml has the hyperparameter changes
```

Hyperparameters can also be tuned by using a genetic algorithm. Instructions for this is also given in the [docs](https://docs.ultralytics.com/guides/hyperparameter-tuning/). This method was not used originally, but can be used to improve hyperparameters.

More detailed documentation about how the YOLO model works and how to train and make predictions can be see in [documentation.md](documentation.md)

Image cropping to 1024x1024 or any other size can be done with img_split.py. Make sure the image path for training points to the output folder of this tool.

Model weights for pre-trained xView adn Openbuildings Dataset path: ./OE_CL_Processing/pre_trained/weights/best.pt

Transfer learning for customer images done with K-folds cross validation with 5 folds.
K-folds weights for customer data: ./OE_CL_Processing/k_folds_cross_val_m/

The inference can be run on the entire [dataset](main.py) or on any [single image](single_image.py). The visualization of both shows a few things. Green boxes are true positives, red are false negatives or misses, purple boxes are false positives and grey boxes are predictions made on the image outside of the pipeline corridor. The gray boxes can be turned off to make the image cleaner as these are not counted in the metrics. Thinner box lines indicate model predictions, while thicker box lines indicate baseline labels.