# 5.1.1 Underfitting and overfitting
# Listing 5.1 Adding white noise channels or all-zeros channels to MNIST
from tensorflow.keras.datasets import mnist
import numpy as np

(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
print(f"len(train_images): {len(train_images)}")
print(f"len(train_images[0]): {len(train_images[0])}")

train_images_with_noise_channels = np.concatenate(
        [train_images, np.random.random((len(train_images), 784))], axis=1)
print(f"len(train_images_with_noise_channels): {len(train_images_with_noise_channels)}")
print(f"len(train_images_with_noise_channels[0]): {len(train_images_with_noise_channels[0])}")

train_images_with_zeros_channels = np.concatenate(
        [train_images, np.zeros((len(train_images), 784))], axis=1)

# Listing 5.2 Training the same model on MNIST data with nopise channels or all-zeros channels
from tensorflow import keras
from tensorflow.keras import layers

def get_model():
    model = keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
    return model

model = get_model()
history_noise = model.fit(
        train_images_with_noise_channels,
        train_labels,
        epochs=10,
        batch_size=128,
        validation_split=0.2)

model = get_model()
history_zeros = model.fit(
        train_images_with_zeros_channels,
        train_labels,
        epochs=10,
        batch_size=128,
        validation_split=0.2)

# Listing 5.3 Plotting a validation accuracy comparison
"""
noise: loss: 0.0216 - accuracy: 0.9933 - val_loss: 0.1803 - val_accuracy: 0.9571
zeros: loss: 0.0106 - accuracy: 0.9974 - val_loss: 0.0911 - val_accuracy: 0.9783 
"""
import matplotlib.pyplot as plt
val_acc_noise = history_noise.history["val_accuracy"]
val_acc_zeros = history_zeros.history["val_accuracy"]
epochs = range(1, 11)
plt.plot(epochs, val_acc_noise, "b-", label="Val Acc with Noise")
plt.plot(epochs, val_acc_zeros, "b--", label="Val Acc with Zeros")
plt.title("Effect of noise on val accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()




