"""
Create and Manage Library Imports for the Project
This script, libraries.py, is designed to centralize the import of necessary libraries and modules for the project.
The project structure includes a 'Data' folder with a 'Dataset' subfolder, and a 'config.py' file.
It also includes a 'reid' folder containing 'data.py', 'model.py', and 'similarity.py'.
Inside the 'reid' folder, there's a 'runs' folder containing the 'train.ipynb' and 'analysis.ipynb' files.
The 'runs' folder also includes two subfolders: 'D2W0.5' and 'D2W1'.
This script facilitates easy import of libraries across various components of the project by adding the project directories to the system path and importing required libraries and modules.

Usage:
To use this script, simply import it at the beginning of your Python notebooks or scripts:
import libraries

Args:
    None

Returns:
    None
"""
# ----------------------------------
#         Necessary libraries       | 
# ----------------------------------
# To import necessary libraries in any file, use the command below:
#import libraries
import sys
sys.path.append('../..')
import os
import copy
import keras
import keras_cv
import numpy as np
from keras import layers
from pathlib import Path
import tensorflow as tf
from similarity import CosineSimilarity, similarity_loss, similarity_recall
from config import EFFICIENTNET_PRESET, RESNET_PRESET, REID_NAME, MODEL_SAVE_LAST,MODEL_SAVE_BEST
from reid.similarity import similarity_scores
from reid import model, data, similarity
from config import dataset_path
from reid.data import prepare_datasets, ReID_Dataset
from reid.model import create_model 
import seaborn as sns
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

