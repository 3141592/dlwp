# 7.3.3 Writing your own callbacks

#
# Listing 7.17 The standard workflow: compile(), fit(), evaluate(), predict()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import keras.backend as K

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

#
# Listing 7.20 Creating a custom callback by subclassing the Callback class
from matplotlib import pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(range(len(self.per_batch_losses)),
                self.per_batch_losses,
                label="Training loss for each batch")
        plt.xlabel(f"Batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plot_as_epoch_{epoch}")
        self.per_batch_losses = []

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
        epochs=10,
        callbacks=[LossHistory()],
        validation_data=(val_images, val_labels))


