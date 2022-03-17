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
print("last_conv_layer_model layers")
for layer in last_conv_layer_model.layers:
    print(f"Layer: {layer.name}")

for layer_name in classifier_layer_names:
    x = model.get_layer(layer_name)(x)
classifier_model = keras.Model(classifier_input, x)

