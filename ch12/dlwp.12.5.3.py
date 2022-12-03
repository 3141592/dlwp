# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("Listing 12.31 Creating a dataset from a directory of images")
from tensorflow import keras
dataset = keras.utils.image_dataset_from_directory(
    "/root/src/data/celeba_gan/",
    # Only the images will be returned--no labels.
    label_mode=None,
    image_size=(64, 64),
    batch_size=32,
    # We will resize the images to 64 x 64 by using a smart combination
    # of cropping and resizing to preserve aspect ratio.
    # We don't want face proportions to get distorted!
    smart_resize=True)

print("Listing 12.32 Rescaling the images")
dataset = dataset.map(lambda x: x / 255.)

print("Listing 12.33 Displaying the first image")
import matplotlib.pyplot as plt
for x in dataset:
    plt.axis("off")
    plt.imshow((x.numpy() * 255).astype("int32")[0])
    plt.show()
    break

print("Listing 12.34 The GAN discriminator network")
from tensorflow.keras import layers

discriminator = keras.Sequential(
        [
            keras.Input(shape=(64, 64, 3)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            # One dropout layer: an important trick!
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )

discriminator.summary()

print("Listing 12.35 GAN generator network")
# The latent space will be made of 128-dimensional vectors.
latent_dim = 128

generator = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            # Produce the same number of coefficients we had at 
            # the level of the Flatten layer in the encoder.
            layers.Dense(8 * 8 * 128),
            # Revert the Flatten layer of the encoder.
            layers.Reshape((8, 8, 128)),
            # Revert the Conv2D layers of the encoder.
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            # We use LeakyReLU as our activation.
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ],
        name="generator",
    )

generator.summary()



