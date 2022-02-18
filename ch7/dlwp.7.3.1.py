# 7.3 Using built-in training and evaluation loops

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

#
# Listing 7.18 Implementing a custom metric by subclassing the Metric class
import tensorflow as tf

# Subclass the Metric class.
class RootMeanSquaredError(keras.metrics.Metric):
    # Define the state variables in the constructor.
    # Like for Layers, you have access to the add_weight() method.
    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
        print(f"self.mse_sum.__class__.__name__: {self.mse_sum.__class__.__name__}")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype="int32")

    # Implement the state update logic in update_state(). The y_true argument
    # is the targets (or labels) for one batch, while y_pred represents the
    # corresponding predictions from the model. You can ignore the
    # sample_weight argument--we won't use it here.
    def update_state(self, y_true, y_pred, sample_weight=None):
        # To match our MNIST model, we expect actegorical predictions and integer labels.
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        #K.print_tensor(mse, message='mse = ')
        #K.print_tensor(self.mse_sum, message='self.mse_sum = ')
        num_samples = tf.shape(y_pred)[0]
        #K.print_tensor(num_samples, message='num_samples = ')
        self.total_samples .assign_add(num_samples)

    def result(self):
        result = tf.sqrt(self.mse_sum / tf.cast(self.total_samples, tf.float32))
        #K.print_tensor(result, message='result = ')
        return result

    def reset_state(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0)

print("New mnist model with custom metric.")
model = get_mnist_model()
model.compile(optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", RootMeanSquaredError()])
model.fit(train_images,
        train_labels,
        epochs=3,
        validation_data=(val_images, val_labels))
test_metrics = model.evaluate(test_images, test_labels)




