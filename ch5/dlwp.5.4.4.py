# 5.4.4 Regularizing your model
# Listing 5.10 Original model
from tensorflow.keras.datasets import imdb
(train_data, train_labels), _ = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):

