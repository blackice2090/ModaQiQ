import keras
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


@keras.saving.register_keras_serializable("similarity", "CosineSimilarity")
class Layer(keras.layers.Layer):
    def call(self, inputs):
        # inputs = keras.utils.normalize(inputs)
        return inputs @ keras.ops.swapaxes(inputs, -2, -1)


def targets(labels):
    targets = labels[None, :] == labels[:, None]
    targets = keras.ops.cast(targets, "float32") + 1
    targets = keras.ops.tril(targets, -1) - 1
    return targets


@keras.saving.register_keras_serializable("similarity")
def loss(true, pred, weight=1):
    true = targets(true)
    mask = true != -1
    true, pred = true[mask], pred[mask]
    # loss = keras.ops.binary_crossentropy(true, pred * 0.5 + 0.5)
    loss = keras.ops.binary_crossentropy(true, pred, from_logits=True)
    if weight != 1:
        loss *= (weight - 1) * true + 1
    return keras.ops.mean(loss, axis=-1)


@keras.saving.register_keras_serializable("similarity")
def recall(true, pred, threshold=0.5):
    true = targets(true)
    mask = true != -1
    true, pred = true[mask], pred[mask]
    positive = pred > threshold
    return keras.ops.mean(positive[true == 1], axis=-1)


def scores(model, dataset):
    out_same = []
    out_diff = []
    for images, labels in tqdm(dataset):
        labels = targets(labels)
        similarity = model(images).numpy()
        same = np.where(labels == 1, similarity, -1)
        diff = np.where(labels == 0, similarity, -1)
        out_same.append(same[same != -1])
        out_diff.append(diff[diff != -1])
    return (np.concatenate(out_same), np.concatenate(out_diff))


def plot(same_scores, diff_scores, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(same_scores, bins=30, alpha=0.7, color="blue", label="Similar")
    ax.hist(diff_scores, bins=30, alpha=0.7, color="red", label="Different")
    ax.set_title("Histogram of Similarity Scores")
    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    return ax


def plot_hard(images, scores, pairs):
    images = images.astype("uint8")
    pairs = pairs[np.random.permutation(len(pairs))]
    fig, axes = plt.subplots(2, min(len(pairs), 6), figsize=(10, 3), dpi=150)
    for ax, pair in zip(np.atleast_2d(axes.T), pairs):
        pair = tuple(pair)
        ax[0].imshow(images[pair[0]])
        ax[1].imshow(images[pair[1]])
        ax[0].set_title(f"{scores[pair]:.2%}")
        ax[0].axis("off")
        ax[1].axis("off")
    fig.tight_layout()
    return axes
