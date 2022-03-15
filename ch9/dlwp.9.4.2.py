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
    activation = feature_extractor(image)
    # Note that we avoid border artifacts by only involving non-border
    # pixels in the loss; we discard the first two pixels along the sides
    # of the activations.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    # Return the mean of the activation values for the filter.
    return tf.reduce_mean(filter_activation)

#
# Listing 9.16 Loss maximization via stochastic gradient ascent
print("Listing 9.16 Loss maximization via stochastic gradient ascent")

@tf.function
def gradient_ascent_step(image, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        # Explicitly watch the image tensor, since it isn't a TensorFlow Variable
        # (only Variables are automatically watched in a GradientTape).
        tape.watch(image)
        # Compute the loss scalar, indicating how much the 
        # current filter activates the filter.
        loss = compute_loss(image, filter_index)
    # Compute the gradients of the loss with respect to the image.
    grads = tape.gradient(loss, image)
    #Apply the "gradient normaliztion trick."
    grads = tf.math.l2_normalize(grads)
    # Now move the image a little bit in a direction that activates our target filter more strongly.
    image += learning_rate * grads
    # Return the updated image so we can run the step function in a loop.
    return image

#
# Listing 9.17 Function to generate filter visualizations
img_width = 200
img_height = 200

def generate_filter_pattern(filter_index):
    # Number of gradient ascent steps to apply
    iterations = 30
    # Amplitude of a single step
    learning_rate = 10.
    # Initialize an image tensor with random values (the Xceptiion model
    # expects input values in the [0, 1] range, so here we pick a range
    # centered on 0.5).
    image = tf.random.uniform(
            minval=0.4,
            maxval=0.6,
            shape=(1, img_width, img_height, 3))
    # Repeatedly update the values of the image tensor so as to maximize our loss function.
    for i in range(iterations):
        image = gradient_ascent_step(image, filter_index, learning_rate)
    return image[0].numpy()

#
# Listing 9.18 Utility function to convert a tensor into a valid image
print("Listing 9.18 Utility function to convert a tensor into a valid image")
def deprocess_image(image):
    # Normalize image values within the [0, 255] range.
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    image = np.clip(image, 0, 255).astype("uint8")
    image = image[25:-25, 25:-25, :]
    return image

print("Figure 9.16")
import matplotlib.pyplot as plt
plt.axis("off")
plt.imshow(deprocess_image(generate_filter_pattern(filter_index=2)))
plt.show()

#
# Listing 9.19 Generating a grid of all filter response patterns in a layer
print("Listing 9.19 Generating a grid of all filter response patterns in a layer")
# Generate and save visualizations for the first 64 filters in the layer.
all_images = []
for filter_index in range(64):
    print(f"Processing filter {filter_index}")
    image = deprocess_image(
            generate_filter_pattern(filter_index)
    )
    all_images.append(image)

# Prepare a blank canvas for us to paste filter visualization on.
margin = 5
n = 8
cropped_width = img_width - 25 * 2
cropped_height = img_height - 25 * 2
width = n * cropped_width + (n - 1) * margin
height = n * cropped_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# Fill the picture with the saved filters.
for i in range(n):
    for j in range(n):
        image = all_images[i * n + j]
        stitched_filters[
                (cropped_width + margin) * i :  (cropped_width + margin) * i + cropped_width,
                (cropped_height + margin) * j : (cropped_height + margin) * j + cropped_height,
                :, 
        ] = image

# Save the canvas to disk.
keras.utils.save_img(f"filters_for_layer_{layer_name}.png", stitched_filters)

plt.axis("off")
plt.imshow(stitched_filters)
plt.show()



