import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize images to [0, 1] range and add a channel dimension
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Create batches and shuffle the dataset
BUFFER_SIZE = 60000
BATCH_SIZE = 128

dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

# Load MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize images to [0, 1] range and add a channel dimension
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Create batches and shuffle the dataset
BUFFER_SIZE = 60000
BATCH_SIZE = 128

dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)


def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(7 * 7 * 192, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((7, 7, 192)))
    model.add(layers.Dropout(0.4))

    model.add(layers.UpSampling2D())
    model.add(
        layers.Conv2DTranspose(
            96, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.UpSampling2D())
    model.add(
        layers.Conv2DTranspose(
            48, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            24, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            1,
            (5, 5),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            activation="sigmoid",
        )
    )
    return model


generator = build_generator()


def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(7 * 7 * 192, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((7, 7, 192)))
    model.add(layers.Dropout(0.4))

    model.add(layers.UpSampling2D())
    model.add(
        layers.Conv2DTranspose(
            96, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.UpSampling2D())
    model.add(
        layers.Conv2DTranspose(
            48, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            24, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            1,
            (5, 5),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            activation="sigmoid",
        )
    )
    return model


generator = build_generator()


def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(7 * 7 * 192, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((7, 7, 192)))
    model.add(layers.Dropout(0.4))

    model.add(layers.UpSampling2D())
    model.add(
        layers.Conv2DTranspose(
            96, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.UpSampling2D())
    model.add(
        layers.Conv2DTranspose(
            48, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            24, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            1,
            (5, 5),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            activation="sigmoid",
        )
    )
    return model


generator = build_generator()


from IPython import display


def generate_and_save_images(model, epoch, test_input):
    # Notice training=False
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap="gray")
        plt.axis("off")

    plt.savefig(f"./images/image_at_epoch_{epoch:04d}.png")
    plt.show()


EPOCHS = 50

train(dataset, EPOCHS)
