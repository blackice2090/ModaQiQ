"""
Model Construction Script
This script outlines the process for constructing deep learning models
using EfficientNet and ResNet backbones,
integrating a custom CosineSimilarity layer for feature comparison.
It demonstrates how to select model backbones
based on configuration presets and compile the model
 with a custom loss function and recall metric tailored for similarity tasks.

Features:
- Initializes EfficientNet and ResNet models based on presets ('EFFICIENTNET_PRESET', 'RESNET_PRESET') specified in 'config.py'.
- Defines a function 'create_model' to build a sequential model incorporating the selected backbone, followed by global average pooling and a CosineSimilarity layer.
- Compiles the model with a 'similarity_loss' function, using 'Adam' optimizer and tracking 'similarity_recall' as a performance metric.

Usage:
- Import necessary libraries and modules, including keras, keras_cv, and custom similarity functions.
- Execute the script directly to create a model instance and display its summary, facilitating immediate insight into the model's architecture and parameters.

"""

import keras
import keras_cv
from keras import layers
from reid.similarity import CosineSimilarity, similarity_loss, similarity_recall, similarity_scores
from config import RESNET_PRESET,EFFICIENTNET_PRESET
from reid.data import ReID_Dataset

# Backbone WEIGHT to None
Backbone_weights = None

# ----------------------------------
#        EfficientNet model         | 
# ----------------------------------
# Create an EfficientNet model instance
# efficientnet_model = keras_cv.models.EfficientNetV2B0Backbone.from_preset(EFFICIENTNET_PRESET)
#EFFICIENTNET_PRESET = keras.applications.EfficientNetB2(EFFICIENTNET_PRESET,include_top=False, weights=Backbone_weights)
# ----------------------------------
#        ResNet model               | 
# ----------------------------------
# Create an ResNet model instance
#resnet_model = keras_cv.models.ResNetBackbone.from_preset(RESNET_PRESET)
RESNET_PRESET = keras.applications.ResNet50(include_top=False, weights=None)
#RESNET_PRESET = keras.applications.ResNet50(include_top=False, weights='imagenet')

# ----------------------------------
#       Model Building Section      | 
# ----------------------------------
# Code for model building
def create_model():
    keras.utils.clear_session() # Clearing the session
    model = keras.Sequential(
        [
            keras.Input(shape=(256, 256, 3)),
            RESNET_PRESET,
            keras.layers.GlobalAvgPool2D(),
            # Assuming CosineSimilarity is a defined layer for computing cosine similarity between features
            CosineSimilarity(),
        ]
    )
    # Learning rate
    learning_rate = 0.001

    model.compile(
        loss=similarity_loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[similarity_recall,]
    )

    return model
if __name__ == "__main__":
    model = create_model()
    model.summary()  # Displaying the model summary
