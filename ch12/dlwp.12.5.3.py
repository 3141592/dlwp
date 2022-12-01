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

