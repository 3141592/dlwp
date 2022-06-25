# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("12.1.4 Implementing text generation with Keras")
print("Creating a database from text files (one file = one sample")
import tensorflow as tf
from tensorflow import keras
dataset = keras.utils.text_dataset_from_directory(
        directory="/root/src/data/aclImdb",
        label_mode=None,
        batch_size=256)

# Strip the <br /> HTML tag that occurs in many of the reviews.
dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br />", " "))
print("list(dataset)[0]:")
print(list(dataset)[0])

print("Listing 12.4 Preparing a TextVectorization layer")

