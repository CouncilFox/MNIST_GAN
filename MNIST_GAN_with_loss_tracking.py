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


# Discriminator model
def build_discriminator():
    model = models.Sequential(
        [
            layers.Conv2D(
                32, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]
            ),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.4),
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.4),
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


# Generator model
def build_generator():
    model = models.Sequential(
        [
            layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(
                128, (5, 5), strides=(1, 1), padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                64, (5, 5), strides=(2, 2), padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                1,
                (5, 5),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                activation="tanh",
            ),
        ]
    )
    return model


# Loss functions and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = optimizers.Adam(1e-4)
discriminator_optimizer = optimizers.Adam(1e-4)

# Create models
generator = build_generator()
discriminator = build_discriminator()

# Training loop
discriminator_losses = []
adversarial_losses = []

EPOCHS = 50
seed = tf.random.normal([16, 100])  # Fixed seed for visualization

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    for step, real_images in enumerate(dataset):
        print(f"Epoch {epoch + 1}, Step {step + 1}/{len(dataset)}")

        # Generate random noise and create fake images
        noise = tf.random.normal([BATCH_SIZE, 100])
        fake_images = generator(noise)

        # Train the discriminator
        with tf.GradientTape() as disc_tape:
            real_output = discriminator(real_images, training=True)
            fake_output = discriminator(fake_images, training=True)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        )

        # Train the generator (via the adversarial model)
        with tf.GradientTape() as gen_tape:
            fake_images = generator(noise, training=True)
            fake_output = discriminator(fake_images, training=True)
            gen_loss = generator_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )
        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )

        # Append losses for tracking
        discriminator_losses.append(disc_loss.numpy())
        adversarial_losses.append(gen_loss.numpy())

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(discriminator_losses, label="Discriminator Loss")
plt.plot(adversarial_losses, label="Adversarial Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("GAN Losses Over Training")
plt.show()
