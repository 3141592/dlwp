# 7.4.3 A complete training and evaluation loop
# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
# Listing 7.21 Writing a step-by-step training loop: the training step function
model = get_mnist_model()

# Prepare the loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy()
# Prepare the optimizer
optimizer = keras.optimizers.RMSprop()
# Prepare the list of metrics to monitor.
metrics = [keras.metrics.SparseCategoricalAccuracy()]
# Prepare a Mean metric tracker to keep track of the avergae loss.
loss_tracking_metric = keras.metrics.Mean()

def train_step(inputs, targets):
    # Run the forward pass. Note that we pass training=True.
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    # Run the backward pass. Note that we use model.trainable_weights.
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # Keep track of metrics.
    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs[metric.name] = metric.result()

    # Keep track of the loss average.
    loss_tracking_metric.update_state(loss)
    logs["loss"] = loss_tracking_metric.result()
    # Return the current values of the metrics and the loss.
    return logs

# Listing 7.22 Writinn a step-by-step training loop: restting the metrics
def reset_metrics():
    for metric in metrics:
        metric.reset_state()
    loss_tracking_metric.reset_state()



model.summary()


