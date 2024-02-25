"""
This script defines a custom CosineSimilarity layer, along with functions for calculating similarity targets, similarity loss, similarity recall, and similarity scores for a dataset. It is designed for use in models that require an understanding of the similarity between different inputs.

- CosineSimilarity: A keras layer that computes cosine similarity between samples in its input.
- similarity_targets: Generates a matrix indicating if pairs of samples are similar (1) or different (0), adjusted to avoid negative similarity values.
- similarity_loss: A custom loss function that calculates binary cross-entropy between true labels and predictions, with an option to weight positive samples more heavily.
- similarity_recall: Computes the recall for positive samples based on a similarity threshold, excluding samples marked with -1.
- similarity_scores: Evaluates a model on a dataset, returning similarity scores for "same" and "different" image pairs, facilitating model evaluation and tuning.

"""
 
import keras
import numpy as np
from config import WEIGHT

# ----------------------------------
# Building custom layer and Function| 
# ----------------------------------
# Calculate Similarity Scores-targets-Cosine...
@keras.saving.register_keras_serializable()
class CosineSimilarity(keras.layers.Layer):
    # A custom layer to compute cosine similarity between samples in the input.
    def call(self, inputs):
    # Normalize the inputs to ensure each input vector has unit length.
        inputs = keras.utils.normalize(inputs)
    # Compute the dot product of inputs with themselves, resulting in a similarity matrix.
        return inputs @ keras.ops.swapaxes(inputs, -2, -1)

def similarity_targets(images, labels):
    # This function creates similarity targets based on labels.
    # It generates a boolean matrix indicating whether labels match (True) or not (False).
    targets = labels[None, :] == labels[:, None]
    # Convert boolean values to float32 and add 1 to avoid negative similarity values.
    targets = keras.ops.cast(targets, "float32") + 1
    # Use the lower part of the matrix to remove duplicate and self comparisons, then subtract 1 to revert values to the original range.
    targets = keras.ops.tril(targets, -1) - 1
    return images, targets

@keras.saving.register_keras_serializable()
def similarity_loss(true, pred, weight=WEIGHT): #Set the WEIGHT from config.py file.
    # A loss function that calculates the similarity loss between true and predicted values.
    loss = keras.ops.binary_crossentropy(true, pred * 0.5 + 0.5)
    # Apply a weight to the loss to increase the impact of positive samples.
    if weight != 1:
        loss *= (weight - 1) * true + 1
    # Create a mask to exclude samples with a value of -1 from the loss calculation.
    mask = keras.ops.cast(true != -1, loss.dtype)
    # Compute the mean loss, considering the mask to exclude some values.
    return keras.ops.mean(loss * mask, axis=-1)

@keras.saving.register_keras_serializable()
def similarity_recall(true, pred, threshold=0.5):
    # A function to compute recall based on a specified threshold.
    mask = true != -1
    # Apply the mask to exclude samples with a value of -1.
    true, pred = true[mask], pred[mask]
    # Determine which samples are considered positive based on the threshold.
    positive = pred > threshold
    # Calculate the mean of correct positive values based on true values.
    return keras.ops.mean(positive[true == 1], axis=-1)

def similarity_scores(model, dataset):
    # Initialize two empty lists to hold similarity scores for "same" and "different" image pairs.
    out_same = [] 
    out_diff = [] 
    
    # Iterate over each batch of images and their corresponding labels in the dataset.
    for images, labels in dataset:
        # Use the model to predict similarity scores for the batch of image pairs.
        similarity = model(images)
        
        # Identify scores where the label is 1 (images are similar) and set others to -1.
        # This filters out scores for different image pairs, keeping only the scores for similar pairs.
        same = np.where(labels == 1, similarity, -1)
        
        # Similarly, identify scores where the label is 0 (images are different) and set others to -1.
        # This filters out scores for similar image pairs, keeping only the scores for different pairs.
        diff = np.where(labels == 0, similarity, -1)
        
        # Append the filtered scores to the respective lists, excluding any -1 values.
        # This step effectively collects similarity scores for all similar pairs and different pairs across the entire dataset.
        out_same.append(same[same != -1])
        out_diff.append(diff[diff != -1])
    
    # Concatenate all collected scores into two arrays: one for similar pairs and one for different pairs.
    # These arrays are then returned as the output of the function.
    return (np.concatenate(out_same), np.concatenate(out_diff))
