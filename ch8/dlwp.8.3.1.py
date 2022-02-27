# 8.3.1 Feature extraction with a pretrained model
# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 8.2.2
new_base_dir = pathlib.Path("/root/src/cats_vs_dogs_small")

#
# Listing 8.9 Using image_dataset_from_directory to read images
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

train_dataset = image_dataset_from_directory(
        new_base_dir / "train",
        image_size=(180, 180),
        batch_size=32)
validation_dataset = image_dataset_from_directory(
        new_base_dir / "validation",
        image_size=(180, 180),
        batch_size=32)
test_dataset = image_dataset_from_directory(
        new_base_dir / "test",
        image_size=(180, 180),
        batch_size=32)

#
# Listing 8.19 Instantiating the VGG16 convolutional base
conv_base = keras.applications.vgg16.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(180, 180, 3))

conv_base.summary()


