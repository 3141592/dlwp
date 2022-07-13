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

print("")
print("Listing 12.4 Preparing a TextVectorization layer")
from tensorflow.keras.layers import TextVectorization

sequence_length = 100
# We'll only consider the 15,000 most common words--
# anything else will be treated as out-of-vocabulary token, "[UNK]".
vocab_size = 15000
text_vectorization = TextVectorization(
        max_tokens=vocab_size,
        # We want to return integer word index sequences.
        output_mode="int",
        # We'll work with inputs and targets of length 100
        # (but since we'll offset the targets by 1, the model will actually see sequences of length 99).
        output_sequence_length=sequence_length,
)
text_vectorization.adapt(dataset)

print("Listing 12.5 Setting up a language modeling dataset")
def prepare_lm_dataset(text_batch):
    # Convert a bunch of texts to a batch of integer sequences.
    vectorized_sequences = text_vectorization(text_batch)
    # Create inputs by cutting off the last word of the sequences.
    x = vectorized_sequences[:, :-1]
    # Create targets by offsetting the sequences by 1.
    y = vectorized_sequences[:, 1:]
    return x, y

lm_dataset = dataset.map(prepare_lm_dataset, num_parallel_calls=4)

