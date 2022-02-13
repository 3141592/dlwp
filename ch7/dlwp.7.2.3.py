# 7.2.3 Subclassing the Model class
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf

#
# Listing 7.9 A multi-input, multi-output Functional model
num_samples = 1280
vocabulary_size = 10000
num_tags = 10
num_departments = 4

# Define model inputs
title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

# Dummy input data
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# Dummy target data
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.random(size=(num_samples, num_departments))

# Listing 7.14 A simple subclassed model
class CustomerTicketModel(keras.Model):

    def __init__(self, num_departments):
        # Call the super() constructor!
        super().__init__()
        # Define sublayers in the constructor
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(num_departments, activation="softmax")

    # Define the forward pass in the call() method
    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)

        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department

model = CustomerTicketModel(num_departments=4)

priority, department = model(
        {"title": title_data,
         "text_body": text_body_data,
         "tags": tags_data})

print("About to model.compile()")
model.compile(optimizer="rmsprop",
        # The structure of what you pass as the loss and
        # metrics arguments must match exactly what gets 
        # returned by call()--here, a list of two elements.
        loss=[["mean_squared_error"], ["categorical_crossentropy"]],
        metrics=[["mean_absolute_error"],["accuracy"]])
# The structure of the input data must match 
# exactly what is expected by the call() method--
# here, a dict with keys title, text_body, tags.
print("About to model.fit()")
model.fit({"title": title_data,
    "text_body": text_body_data,
    "tags": tags_data},
    # The structure of the target
    # data must match exactly what is returned by the call() method--
    # here, a list of two elements.
    [priority_data, department_data],
    epochs=1)
print("About to model.evaluate()")
model.evaluate({"title": title_data,
    "text_body": text_body_data,
    "tags": tags_data},
    [priority_data, department_data])
print("About to model.predict()")
priority_preds, department_preds = model.predict({"title": title_data,
                                                  "text_body": text_body_data,
                                                  "tags": tags_data})

print(f"tf.keras.backend.max(priority_preds): {tf.keras.backend.max(priority_preds)}")
print(f"tf.keras.backend.max(department_preds): {tf.keras.backend.max(department_preds)}")

model.summary()
keras.utils.plot_model(model,
                "7.2.3.simple_subclass_model.png",
                 show_shapes=True)

