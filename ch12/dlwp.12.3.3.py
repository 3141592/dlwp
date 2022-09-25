# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("12.3.3 Nerual style transfer in Keras")

print("Listing 12.16 Getting the style and content images")
import tensorflow as tf
from tensorflow import keras

base_image_path = keras.utils.get_file("sf.jpg", origin="https://img-datasets.s3.amazonaws.com/sf.jpg")
style_refernce_image_path = keras.utils.get_file("starry_might.jpg", origin="https://img-datasets.s3.amazonaws.com/starry_night.jpg")

original_width, original_height = keras.utils.load_img(base_image_path).size
img_height = 400
img_width = round(original_width * img_height / original_height)

print("Listing 12.17 Auxillary functions")
import numpy as np

# Util functions to open, resize, and format pictures into appropriate arrays
def preprocess_image(image_path):
    img = keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img = keras.utils.image_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg19.preprocess_input(img)
    return img

# Util function to convert a NumPy array into a valid image
def deprocess_image(img):
    img = img.reshape((img_height, img_width, 3))
    # Zero-centering by removing the mean pixel value from ImageNet.
    # This reverses a transformation done by vgg19.preprocess_input.
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # Converts images from 'BGR' to 'RGB'.
    # This is also part of the reversal of vgg19.preprocess_input.
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astytpe("uint8")
    return img

print("Listing 12.18 Using a pretrained VGG19 model to create a feature extractor")
# Build a VGG19 model loaded with pretrained ImageNet weights.


