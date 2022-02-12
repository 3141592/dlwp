# 7.2.1 The Sequential model
# Listing 7.1 The Sequential class
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Listing 7.2 Incrementally building a Sequential model
model = keras.Sequential()
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

# Listing 7.3 Models that aren't yet built have no weights
try:
    print(f"model.weights: {model.weights}")

except:
    pass

# Listing 7.4 Calling a model for the first time to build it
model.build(input_shape=(None, 3))
print(f"model.weights: {model.weights}")
model.summary()

# Listing 7.6 Naming models and layers with the name argument
model = keras.Sequential(name="my_example_model")
model.add(layers.Dense(64, activation="relu", name="my_first_layer"))
model.add(layers.Dense(10, activation="softmax", name="my_last_layer"))

model.build(input_shape=(None, 3))
model.summary()

# Listing 7.7 Specifying the input shape of your model in advance
model = keras.Sequential(name="my_example_model")
model.add(keras.Input(shape=(3, 1)))
print("No layers...")
model.summary()
model.add(layers.Dense(64, activation="relu", name="my_first_layer"))
print("My first layer...")
model.summary()
model.add(layers.Dense(10, activation="softmax", name="my_last_layer"))
print("My second layer...")
model.summary()

model.build()
print("After model.build()...")
model.summary()

keras.utils.plot_model(model, "functional_api_7_2_1.png")


