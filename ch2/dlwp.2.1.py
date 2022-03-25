# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Loading the MNIST dataset in Keras
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images,test_labels) = mnist.load_data()

print(train_images.shape)
print(len(train_labels))

print(test_images.shape)
print(len(test_labels))

print(test_images[0].shape)
print(test_images[0][7])
print(test_images[0])

# Listing 2.2 The network architecture
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Listing 2.3 The compilation step
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Listing 2.4 Preparing data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# Listing 2.5 "Fitting" the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Listing 2.6 Using the model to make predictions
test_digits = test_images[0:10]
predictions = model.predict(test_digits)

print(predictions[0])
print(predictions[0].argmax())
print(predictions[0][7])
print(test_labels[0])

# Listing 2.7 Evaluating the model on new data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")

