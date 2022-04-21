# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("11.5 Beyond text classification: Sequence-to-sequence learning")
print("11.5.1 A machine translation example")
print("Parse the file")
text_file = "/root/src/spa-eng/spa.txt"
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []

# Iterate over the file in lines
for line in lines:
    # Each line contains an English phrase and its Spanish translation, tab-separated.
    english, spanish = line.split("\t")
    # We prepend "[start]" and append "[end]" to the Spanish sentence, to match the template from figure 11.12.
    spanish = "[start] " + spanish + " [end]"
    text_pairs.append((english, spanish))

print("Shuffle and split data into training, validation, and test")
import random
random.shuffle(text_pairs)

print(random.choice(text_pairs))

print("Listing 11.6 Vectorizing the English and Spanish text pairs")
import tensorflow as tf
import string
import re

# Hold ALT+168 from keypad
strip_chars = string.punctuation + "¿"
strip_chars = strip_char.replace("[", "")
strip_chars = strip_char.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return if tf.strings.regex_replace(
            lowercase, f"[{re.escape(strip_chars)}]", "")

# To keep things simple, we'll only look at the top 15,000 words in each
# language, and we'll restrict sentences to 20 words.
vocab_size = 15000
sequence_length = 20

# The English layer
source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
)
# The Spanish layer
target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        # Generate Spanish sentences that have one extra token,
        # since we'll need to offset the sentence by one step during training.
        output_sequence_length=sequence_length + 1,
        standardize=custom_standardization
)
train_english_texts = [pair[0] 

