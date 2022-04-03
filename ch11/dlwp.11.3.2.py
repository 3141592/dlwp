# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Listing 11.2 Displaying the shapes and dtypes of the first batch")
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
batch_size = 32

train_ds = keras.utils.text_dataset_from_directory(
        "/root/src/aclImdb/train/", batch_size=batch_size)

val_ds = keras.utils.text_dataset_from_directory(
        "/root/src/aclImdb/val/", batch_size=batch_size)

test_ds = keras.utils.text_dataset_from_directory(
        "/root/src/aclImdb/test/", batch_size=batch_size)

for inputs, targets in train_ds:
    print("inputs.shape: ", inputs.shape)
    print("inputs.dtype: ", inputs.dtype)
    print("targets.shape: ", targets.shape)
    print("targets.dtype: ", targets.dtype)
    print("inputs[0]: ", inputs[0])
    print("targets[0]: ", targets[0])
    break

print("11.3.2 Processing words as a set: The bag-of-words approach")
print("Listing 11.3 Preprocessing our datasets with a TextVectorization layer")
text_vectorization = TextVectorization(
        # Limit vocabulary to the 20,000 most frequent words.
        max_tokens=20000,
        # Encode the output tokens as multi-hot binary vectors.
        output_mode="multi_hot")
# Prepare a dataset that only yeilds raw text inputs (no labels).
text_only_train_ds = train_ds.map(lambda x, y: x)

