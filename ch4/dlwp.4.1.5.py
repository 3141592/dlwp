# Listing 4.1 Loading the IMDB dataset
from tensorflow.keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Listing 4.3 Encoding the integer sequences via multi-hot encoding
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")
print(f"x_train.shape: {x_train.shape}")
print(f"y_train.shape: {y_train.shape}")

# Listing 4.4 Model definition
from tensorflow import keras
from tensorflow.keras import layers

# Listing 4.10 Retraining a model from scratch
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
    #layers.Dense(1, activation="softmax")
])

model.compile(optimizer="rmsprop",
        loss="binary_crossentropy",
        #loss="mse",
        metrics=["accuracy"])

model.fit(x_train,
        y_train,
        epochs=4,
        batch_size=512)

print("About to model.evaluate()")
results = model.evaluate(x_test, y_test)
print(f"results: {results}")

# 4.1.5 Using a trained model to generate predictions on new data
print("About to model.predict()")
predictions = model.predict(x_test)
print(f"predictions: {predictions}")

"""
4.1.6 Further experiments

Layers Accuracy
binary_crossentropy
relu/sigmoid
2      0.8815199732780457]
1      0.887719988822937

Layers Units Accuracy
binary_crossentropy
relu/sigmoid
1      16
2      16      0.8815199732780457

1       8
2       8      0.8888400197029114
               0.8800399899482727

1      32
2      32      0.880079984664917

1      16
2      16    
3      16      0.8567600250244141

Layers Units Accuracy
binary_crossentropy
tanh/sigmoid
1      16
2      16      0.8774399757385254

Layers Units Accuracy
mse
relu/sigmoid
1      16
2      16      0.8827999830245972 




"""
