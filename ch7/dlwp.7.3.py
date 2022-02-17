# 7.3 Using built-in training and evaluation loops

#
# Listing 7.17 The standard workflow: compile(), fit(), evaluate(), predict()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Create a model (we factor this into a separate function so as to reuse it later).
def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model

# Load your data, reserving some for validation
(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

# Compile the model by specifying its optimizer, 
# the loss function to minimize,
# and the metrics to monitor.
model = get_mnist_model()
model.compile(optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
print("About to model.fit()")
model.summary()

# Use fit() to train the model, optionally providing
# validation data to monitor performance on unseen data.
model.fit(train_images,
        train_labels,
        epochs=3,
        validation_data=(val_images, val_labels))
# Use evaluate() to compute the loss and metrics on new data.
test_metrics = model.evaluate(test_images, test_labels)
# Use predict() to computer classification probabilities on new data.
predictions = model.predict(test_images)

