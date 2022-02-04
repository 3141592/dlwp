# 5.4.4 Regularizing your model
# Listing 5.10 Original model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tqdm.keras import TqdmCallback
import numpy as np

(train_data, train_labels), _ = imdb.load_data(num_words=10000)
print(f"train_data[0]: {train_data[0]}")

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
train_data = vectorize_sequences(train_data)
print(f"train_data[0]: {train_data[0]}")

print("Starting original model")
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"])
history_original = model.fit(
        train_data,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_split=0.4)

# Listing 5.13 Adding L2 weight regularization to the model
from tensorflow.keras import regularizers
print("Starting L2 model")
model = keras.Sequential([
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.002), activation="relu"),
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.002), activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"])
history_l2_reg = model.fit(
        train_data,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_split=0.4)

# Listing 5.15 Adding dropout to the IMDB model
print("Starting dropout model")
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"])
history_dropout = model.fit(
        train_data,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_split=0.4)

# Figure 5.17/5.18
import matplotlib.pyplot as plt

# history_original_model
val_loss = history_original.history["val_loss"]
epochs = range(1, 21)
plt.plot(epochs, val_loss, "b-", label="val loss original")

# history_l2_reg
val_loss = history_l2_reg.history["val_loss"]
plt.plot(epochs, val_loss, "r--", label="val loss l2 reg")

# history_dropout
val_loss = history_dropout.history["val_loss"]
plt.plot(epochs, val_loss, "g--", label="val loss dropout")

plt.title("Original vs. smaller vs. larger vs. l2 reg on IMDB review classification")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
