# 9.4.3 Visualizing heatmaps of class activation
print("9.4.3 Visualizing heatmaps of class activation")
# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 
# Listing 9.20 Loading the Xception network with pretrained weights
print("Listing 9.20 Loading the Xception network with pretrained weights")
import tensorflow.keras as keras
import numpy as np

model = keras.applications.xception.Xception(weights="imagenet")

#
# Listing 9.21 Preprocessing an input image for Xecption
print("Listing 9.21 Preprocessing an input image for Xecption")
img_path = keras.utils.get_file(
        fname="elephant.jpg",
        origin="https://img-datasets.s3.amazonaws.com/elephant.jpg")

def get_img_array(img_path, target_size):
    # Return a Python Imaging Library (PIL) image of size 299 x 299.
    img = keras.utils.load_img(img_path, target_size=target_size)
    # Return a float32 NumPy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # Add a dimension to transform the array into a batch of size (1, 299, 299, 3).
    array = np.expand_dims(array, axis=0)
    # Preprocess the batch (this does channel-wise color normalization).
    array = keras.applications.xception.preprocess_input(array)
    return array

img_array = get_img_array(img_path, target_size=(299, 299))

# Run the pretrained network on the image and deocde its prediction vector
print("Run the pretrained network on the image and deocde its prediction vector")
preds = model.predict(img_array)
print(keras.applications.xception.decode_predictions(preds, top=3)[0])

#
# Listing 9.22 Setting up a model that returns the last convolutional output
print("Listing 9.22 Setting up a model that returns the last convolutional output")
last_conv_layer_name = "block14_sepconv2_act"
classifier_layer_names = [
        "avg_pool",
        "predictions"
        ]
last_conv_layer = model.get_layer(last_conv_layer_name)
last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

#
# Listing 9.23 Reapplying the classifier on top of the last convolutional output
print("Listing 9.23 Reapplying the classifier on top of the last convolutional output")
classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in classifier_layer_names:
    x = model.get_layer(layer_name)(x)
classifier_model = keras.Model(classifier_input, x)

# Listing 9.24 Retrieving the gradients of the top predicted class
print("Listing 9.24 Retrieving the gradients of the top predicted class")
import tensorflow as tf

with tf.GradientTape() as tape:
    # Compute activation of the last conv layer and make the tape watch it.
    last_conv_layer_output = last_conv_layer_model(img_array)
    tape.watch(last_conv_layer_output)
    # Retrieve the activation channel corresponding to the top predicted class
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]

# This is the gradient of the top predicted class with regard
# to the output feature map of the last convolutional layer.
grads = tape.gradient(top_class_channel, last_conv_layer_output)

#
# Listing 9.25 Gradient pooling and channel-importance weighing
print("Listing 9.25 Gradient pooling and channel-importance weighing")

# This is a vector where each entry is the mean intensity of the gradient for a given channel.
# It quantifies the importance of each channel with regard to the top predicted class.
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
last_conv_layer_output = last_conv_layer_output.numpy()[0]

# Multiple each channel in the output of the last convolutional layer
# by "how important this channel is".
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]

# The channel-wise mean of the resulting feature map is our heatmap of class activation.
heatmap = np.mean(last_conv_layer_output, axis=-1)

#
# Listing 9.26 Heatmap post-processing
print("Listing 9.26 Heatmap post-processing")
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

#
# Listing 9.27 Superimposing the heatmap on the original picture
print("Listing 9.27 Superimposing the heatmap on the original picture")
#import matplotlib.pyplot as plt
import matplotlib.cm as cm

img = keras.utils.load_img(img_path)
img = keras.utils.img_to_array(img)

# Rescale the heatmap to 0-255.
print("Rescale the heatmap to 0-255.")
heatmap = np.uint8(255 * heatmap)

# Use the "jet" colormap to recolorize the heatmap.
print("Use the \"jet\" colormap to recolorize the heatmap.")
jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# Create an image that contains the recolorized heatmap
print("Create an image that contains the recolorized heatmap")
jet_heatmap = keras.utils.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.utils.img_to_array(jet_heatmap)

# Superimpose the heatmap and the original image, with the heatmap at 40% opacity.
print("Superimpose the heatmap and the original image, with the heatmap at 40% opacity.")
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.utils.array_to_img(superimposed_img)

print("Saving image.")
save_path = "elephant_cam.jpg"
superimposed_img.save(save_path)
#plt.imshow(superimposed_img)
#plt.show()


