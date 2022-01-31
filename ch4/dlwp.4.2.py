# 4.2.1 The Reuters dataset
# Listing 4.11 Loading the Reuters dataset
from tensorflow.keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print(f"len(train_data): {len(train_data)}")
print(f"len(test_data): {len(test_data)}")

# Listing 4.12 Decoding newswires back to text
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = " ".join(
        [reverse_word_index.get(i - 3, "?") for i in train_data[0]])
print(decoded_newswire)
print(f"train_labels[10]: {train_labels[10]}")

# 4.2.2 Preparing data
# Listing 4.3 Encoding the integer sequences via multi-hot encoding
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        #for j in sequence:
            #results[i, j] = 1.
        results[i, sequence] = 1.
    return results

# Listing 4.13 Encoding the input data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Listing 4.14 Encoding the labels
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

#print(y_train.shape)
#print(y_test.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

print(y_train.shape)
print(y_test.shape)

# 4.2.3 Building your model
from tensorflow import keras
from tensorflow.keras import layers

# Listing 4.15 Model definition
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
    ])

# Listing 4.16 Compiling the model
model.compile(optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"])
        #metrics=["categorical_accuracy"])

# 4.2.4 Validating your approach
# Listing 4.17 Setting aside a validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# Listing 4.18 Training the model
history = model.fit(partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val))

print(history.history.keys())

# Listing 4.19 Printing the training and validation loss
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Trn loss")
plt.plot(epochs, val_loss_values, "b", label="Val loss")
plt.title("Trn and Val Loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
#plt.show()

# Listing 4.20 Plotting the training and validation accuracy
acc = history_dict["accuracy"]
#acc = history_dict["categorical_accuracy"]
val_acc = history_dict["val_accuracy"]
#val_acc = history_dict["val_categorical_accuracy"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, acc, "go", label="Trn acc")
plt.plot(epochs, val_acc, "g", label="Val acc")
plt.title("Trn and Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Listing 4.21 Retraining a model from scratch
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])
model.compile(optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"])
model.fit(x_train,
        y_train,
        epochs=10,
        batch_size=512)
results = model.evaluate(x_test, y_test)
print(f"results: {results}")

# Accuracy of a random baseline
import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
print(f"Random hits_array.mean(): {hits_array.mean()}")


# 4.2.5 Generating predictions on new data
predictions = model.predict(x_test)
print(f"predictions[0].shape: {predictions[0].shape}")
print(f"np.sum(predictions[0]): {np.sum(predictions[0])}")
print(f"np.argmax(predictions[0]): {np.argmax(predictions[0])}")
print(f"predictions[0]: {predictions[0]}")

# 4.2.6 A different way to handle the labels and loss
y_train = np.array(train_labels)
y_test = np.array(test_labels)

model.compile(optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
model.fit(x_train,
        y_train,
        epochs=10,
        batch_size=512)
results = model.evaluate(x_test, y_test)
print(f"results: {results}")

# 4.2.7 The importance of having sufficiently large intermediate layers
# Listing 4.22 A model with an information bottleneck
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(46, activation="softmax")
    ])
model.compile(optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"])
model.fit(partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val))

# Listing 4.19 Printing the training and validation loss
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Trn loss")
plt.plot(epochs, val_loss_values, "b", label="Val loss")
plt.title("Trn and Val Loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
#plt.show()

# Listing 4.20 Plotting the training and validation accuracy
acc = history_dict["accuracy"]
#acc = history_dict["categorical_accuracy"]
val_acc = history_dict["val_accuracy"]
#val_acc = history_dict["val_categorical_accuracy"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, acc, "go", label="Trn acc")
plt.plot(epochs, val_acc, "g", label="Val acc")
plt.title("Trn and Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()






