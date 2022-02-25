# 8.2.5 Using data augmentation
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
# Listing 8.14 Define a data augtmentation stage to add to an image model
data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ]
)

#
# Listing 8.15 Displaying some randomly augmented training images
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
# We can use take(N) to only sample N batches from the dataset.
# This is equivalent to inserting a break in the loop after the Nth batch.
for images, _ in train_dataset.take(1):
    for i in range(9):
        # Apply the augmentationstage to the batch of images.
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        # Display the first image in the output batch.
        # For each of the nine iterations, this is a different augmentation of the same image.
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        
plt.show()

