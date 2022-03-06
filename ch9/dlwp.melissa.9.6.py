# 9.2 An image segmentation example
# Suppress warnings
import os, pathlib
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
from tensorflow.keras.utils import load_img, img_to_array 
import numpy as np
import cv2 as cv

def path_to_input_image(path):
        return img_to_array(load_img(path, target_size=img_size))

#
# Figure 9.6 A test image and its predicted segmentation mask
from tensorflow import keras
from tensorflow.keras.utils import array_to_img
import matplotlib.pyplot as plt

model = keras.models.load_model("oxford.segmentation.50.keras")

i = 4

# We resize everything to 200 x 200
img_size = (200, 200)

test_image = path_to_input_image("/mnt/d/IMG_0486.JPG")
test_image = path_to_input_image("/mnt/d/IMG020.JPG")
test_image = path_to_input_image("/mnt/d/DSCN2287.JPG")

plt.axis("off")
plt.imshow(test_image)
plt.show()

mask = model.predict(np.expand_dims(test_image, 0))[0]

# Utility to display a model's prediction
def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)
    plt.show()

display_mask(mask)

