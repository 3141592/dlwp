# 7.4.5 Leveraging fit() with a custom training loo'
# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime

#
# Listing 7.17 The standard workflow: compile(), fit(), evaluate(), predict()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
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
# Listing 7.26 Implementing a custom training step to use with fit()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
# This metric object will be used to track the average of per-batch losses during training and evaluation.
loss_tracker = keras.metrics.Mean(name="loss")

class CustomModel(keras.Model):
    # We override the train_step method.
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            # We use self(inputs, training=True) instead of model(inputs, training=True), 
            # since our model is the class itself.
            predictions = self(inputs, training=True)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weighs))

        # We update the loss tracker metric that tracks the average of the loss.
        loss_tracker.update_state(loss)
        # We return the average loss so far by querying the loss tracker metric.
        return {"loss": loss_tracker.result()}

    @property
    # Any metric you would like to reset across epochs should be listed here.
    def metrics(self):
        return [loss_tracker]

# We can now instantiate our custom model, compile it (we only pass the optimizer, since
# the loss is already defined outside of the model), and train it using fit() as usual:

