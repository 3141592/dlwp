# 7.2.1 The Functional API
# Listing 7.8 A simple Functional model with two Dense layers
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(3,), name="my_input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs)

print(f"inputs.shape: {inputs.shape}")
print(f"inputs.dtype: {inputs.dtype}")
print(f"features.shape: {features.shape}")
print(f"features.dtype: {features.dtype}")
print(f"outputs.shape: {outputs.shape}")
print(f"outputs.dtype: {outputs.dtype}")
model.summary()

