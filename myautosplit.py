from ultralytics.data.split import autosplit
from pathlib import Path

# assuming a path to images
images = Path('./bhe_data/whole_data')  ##dir with /images /labels
autosplit(images, weights=(0.7,0.15,0.15)) # train, validation, test splits