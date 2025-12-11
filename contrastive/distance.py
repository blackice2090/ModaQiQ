import keras


@keras.saving.register_keras_serializable("distance", "EuclideanDistance")
class Layer(keras.layers.Layer):
    def call(self, inputs):
        return distance(inputs, inputs)


def targets(labels):
    targets = labels[None, :] == labels[:, None]
    targets = keras.ops.cast(targets, "float32") + 1
    targets = keras.ops.tril(targets, -1) - 1
    return targets


@keras.saving.register_keras_serializable("distance")
def loss(true, pred, margin=1, weight=3):
    true = targets(true)
    positive = pred[true == 1]
    negative = keras.ops.relu(margin - pred[true == 0])
    return weight * _mean(positive) + _mean(negative)


def _t(x):
    return keras.ops.swapaxes(x, -2, -1)


def _norm2(x):
    return keras.ops.squeeze(x[:, None] @ x[..., None], -1)

def _mean(x):
    return keras.ops.mean(x, -1)


def distance(x, y):
    out = -2 * (x @ _t(y))
    x_norm = _norm2(x)
    y_norm = x_norm if y is x else _norm2(y)
    return x_norm + out + _t(y_norm)
