# 5.3 Improving model fit
# 5.3.1 Tuning key gradient descent parameters
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np

(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer=keras.optimizers.RMSprop(1.),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
model.fit(train_images,
        train_labels,
        epochs=10,
        batch_size=128,
        validation_split=0.2)
"""
The best result from 50 epochs:
loss: 1.8429 - accuracy: 0.4314 - val_loss: 3.5503 - val_accuracy: 0.4655

Final epoch:
loss: 1.8181 - accuracy: 0.4433 - val_loss: 3.2103 - val_accuracy: 0.4634
"""

# Listing 5.8 The same model with a more appropriate learning rate
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer=keras.optimizers.RMSprop(1e-2),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
model.fit(train_images,
        train_labels,
        epochs=10,
        batch_size=128,
        validation_split=0.2)
"""
Final result:
loss: 0.0581 - accuracy: 0.9899 - val_loss: 0.3332 - val_accuracy: 0.9712
"""

# 5.3.3 Increasing model capacity
# Listing 5.9 A simple logistic regression on MNIST
model = keras.Sequential([layers.Dense(10, activation="softmax")])
model.compile(optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
history_small_model = model.fit(train_images,
        train_labels,
        epochs=20,
        batch_size=128,
        validation_split=0.2)

# Figure 5.14 Effect of insufficient model capacity on loss curve
import matplotlib.pyplot as plt
val_loss = history_small_model.history["val_loss"]
val_acc = history_small_model.history["val_accuracy"]
epochs = range(1, 21)
#plt.plot(epochs, val_acc, "b-", label="val acc")
#plt.plot(epochs, val_loss, "b--", label="val loss")
plt.title("Effect of insufficient model capacity on validation curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
#plt.legend()
#plt.show()

model = keras.Sequential([
    layers.Dense(96, activation="relu"),
    layers.Dense(96, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
history_large_model = model.fit(train_images,
        train_labels,
        epochs=20,
        batch_size=128,
        validation_split=0.2)

# Figure 5.14 Effect of insufficient model capacity on loss curve
import matplotlib.pyplot as plt
val_loss2 = history_large_model.history["val_loss"]
val_acc2 = history_large_model.history["val_accuracy"]
epochs = range(1, 21)
plt.title("Validation loss for a model with appropriate capacity")
#plt.plot(epochs, val_acc2, "g-", label="large acc")
plt.plot(epochs, val_loss2, "g--", label="large loss")
plt.legend()
plt.show()



