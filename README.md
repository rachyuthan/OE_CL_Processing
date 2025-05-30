# OE_CL_Processing
Scripts to run class location and postprocessing 

Run this to setup the environment.
```bash
conda env create -f environment.yaml
conda activate OE_YOLO
```

Run Labels.py to generate truth labels and pull images from server (only works for BHE). 
Run main.py and change directories in the config so that everything is pulled from the correct locations. 
Change the config settings to play around with the confidence threshold for accepting predictions and filtering false positives. 
Run the program.

More detailed documentation about how the YOLO model works and how to train and make predictions can be see in [documentation.md](documentation.md)

Model weights pre-trained path: ./OE_CL_Processing/pre_trained/weights/best.pt

K-folds weights: ./OE_CL_Processing/k_folds_cross_val_m/

The inference can be run on the entire [dataset](main.py) or on any [single image](single_image.py) 