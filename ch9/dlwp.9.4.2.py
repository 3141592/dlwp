# 9.4.2 Visualizing convnet folters
# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Download a test image.
from tensorflow import keras
import numpy as np
img_path = keras.utils.get_file(fname="cat.jpg", origin="https://img-datasets.s3.amazonaws.com/cat.jpg")

def get_img_array(img_path, target_size):
    # Open the image file and resize it.
    img = keras.utils.load_img(img_path, target_size=target_size)

    # Turn the image into a float32 NumPy array of shape (180, 180, 3).
    array = keras.utils.img_to_array(img)
    # Add a dimension to transform the array into a "batch" of a single smaple.
    # It's shape is now (1, 180, 180, 3).
    array = np.expand_dims(array, axis=0)
    return array

img_tensor = get_img_array(img_path, target_size=(180, 180))

#
# Listing 9.12 Instantiating the Xception convolution base
print("Listing 9.12 Instantiating the Xception convolution base")
from tensorflow import keras

model = keras.applications.xception.Xception(
        weights="imagenet",
        # The classification layers are irrelevant for this use case,
        # so we don't include the top stage of the model.
        include_top=False)
model.summary()

#
# Listing 9.13 Printing the names of all convolutional layers in Xception
print("Listing 9.13 Printing the names of all convolutional layers in Xception")
for layer in model.layers:
    if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
        print(layer.name)

#
# Listing 9.14 Creating a feature extractor model
print("Listing 9.14 Creating a feature extractor model")
# You could replace this with the name of any layer in the Xception convolutionalbase.
layer_name = "block3_sepconv1"
# This is the layer object we're interested in.
layer = model.get_layer(name=layer_name)
# We use model.input and layer.output to create a model that,
# given an input image, returns the output of out target layer.
feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)
feature_extractor.summary()

#
# Listing 9.15 Using the feature extractor
print("Listing 9.15 Using the feature extractor")
activation = feature_extractor(keras.applications.xception.preprocess_input(img_tensor))

# Use feature extractor model to define a loss function
print("Use feature extractor model to define a loss function")

import tensorflow as tf

# The loss function takes an image tensor and the index
# of the filter we are considering (an integer).
def compute_loss(image, filter_index):
    activation = feature_extraction(image)
    # Note that we avoid border artifacts by only involving non-border
    # pixels in the loss; we discard the first two pixels along the sides
    # of the activations.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    # Return the mean of the activation values for the filter.
    return tf.reduce_mean(filter_activation)




