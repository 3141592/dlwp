# 7.2.1 The Functional API
#
# Listing 7.8 A simple Functional model with two Dense layers
#
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

#
# Listing 7.9 A multi-input, multi-output Functional model
#
vocabulary_size = 10000
num_tags = 10
num_departments = 4

# Define model inputs
title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

# Combine input features into a single tensor, features, by concatentating them
features = layers.Concatenate()([title, text_body, tags])
# Apply an intermediate layer to recombine input features into richer representations
features = layers.Dense(64, activation="relu")(features)

# Define model outputs
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(num_departments, activation="softmax", name="department")(features)

# Create the model by specifying its inputs and outputs
model = keras.Model(inputs=[title, text_body, tags],
        outputs=[priority, department])

print(f"features.shape: {features.shape}")
model.summary()

#
# Listing 7.10 Training a model by providing lists of input and target arrays
#
import numpy as np

num_samples = 1280

# Dummy input data
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# Dummy target data
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.random(size=(num_samples, num_departments))

model.compile(optimizer="rmsprop",
        loss=["mean_squared_error", "categorical_crossentropy"],
        metrics=[["mean_absolute_error"], ["accuracy"]])
model.fit([title_data, text_body_data, tags_data],
        [priority_data, department_data],
        epochs=1)
model.evaluate([title_data, text_body_data, tags_data],
        [priority_data, department_data])
priority_preds, department_preds = model.predict([title_data, text_body_data, tags_data])

#
# Listing 7.11 Training a model by providing dicts and targets arrays
#
model.compile(optimizer="rmsprop",
        loss={"priority":"mean_squared_error", "department":"categorical_crossentropy"},
        metrics={"priority":["mean_absolute_error"], "department":["accuracy"]})
model.fit({"title": title_data, "text_body": text_body_data, "tags": tags_data},
        {"priority": priority_data, "department": department_data},
        epochs=1)
model.evaluate({"title": title_data, "text_body": text_body_data, "tags": tags_data},
        {"priority": priority_data, "department": department_data})
priority_preds, department_preds = model.predict({"title": title_data, "text_body": text_body_data, "tags": tags_data})











