# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("12.2.1 Implementing DeepDream in Keras")

print("Listing 12.9 Fetching the test image")
from tensorflow import keras
import matplotlib.pyplot as plt

print("Listing 12.10 Instantiating a pretrained InceptionV3 model")
from tensorflow.keras.applications import inception_v3
model = inception_v3.InceptionV3(weights="imagenet", include_top=False)
model.summary()

