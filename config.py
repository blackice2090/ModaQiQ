"""
Configuration Script for Project Setup
This script configures key aspects of the project, including the dataset path, dataset name, model weight, and model save paths. It relies on 'libraries.py' for importing necessary libraries and establishes model type presets for training.

Features:
- 'dataset_path': Defines the location of the dataset for easy access across the project.
- 'REID_NAME': Specifies the name of the ReID dataset used, aiding in dynamic dataset configuration.
- 'WEIGHT': Sets a weight variable that could influence model configuration or performance.
- 'MODEL_SAVE_BEST' and 'MODEL_SAVE_LAST': Determine paths for saving the "best" and "last" model versions, respectively, based on the WEIGHT.
- 'EFFICIENTNET_PRESET' and 'RESNET_PRESET': Define model types for training, allowing for flexible model architecture selection.

"""

import sys
import os
import keras_cv 
from pathlib import Path

# base directory for the project
BASE_DIR = Path("project")

# ----------------------------------
#         Dataset Path              | Here we set the Path
# ----------------------------------
# # Dataset path configuration
dataset_path = "/Users/a1443/tensorflow-test/project/Data/Dataset/bbox_t2"

# ----------------------------------
#        DATASET NAME               | ReID Dataset 1 or 2
# ----------------------------------
# ReID dataset name configuration
# Set the REID_NAME from config.py file
REID_NAME = "ReID_Dataset"  

# ----------------------------------
#      WEIGHT variable              | Here we set the WEIGHT
# ----------------------------------
# Set the WEIGHT in similarity.py file insde similarity_loss Function
WEIGHT = 3

# ----------------------------------
#   Model save path on the WEIGHT   | 
# ----------------------------------
# Model save path best and last configuration 
# In train.ipynb file.
MODEL_SAVE_BEST = '/Users/a1443/tensorflow-test/project/reid/runs/D2W0.5/best_model.keras'
MODEL_SAVE_LAST = '/Users/a1443/tensorflow-test/project/reid/runs/D2W1/last_model.keras'

# ----------------------------------
#        The Model type             | 
# ----------------------------------
# Model type configuration 
# Set in data.py file in side prepare_datasets function 
EFFICIENTNET_PRESET = "efficientnetv2_B2"
RESNET_PRESET = "resnet50"