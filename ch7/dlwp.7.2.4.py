# 7.2.4 Mixing and matching different components
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf

#
# Listing 7.15 Creating a Functional model that includes a subclassed model
class Classifier(keras.Model):

    def __init__(self, num_classes):
        super().__init__()
        if num_classes == 2:
            num_units = 1
            activation = "sigmoid"
        else:
            num_units = num_classes
            activation = "softmax"
        self.dense = layers.Dense(num_units, activation=activation)

    def call(self, inputs):
        return self.dense(inputs)

print("Starting Listing 7.15")
inputs = keras.Input(shape=(3,))
features = layers.Dense(64, activation="relu")(inputs)
outputs = Classifier(num_classes=10)(features)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

#
# Listing 7.16 Creating a subclasses model that includes a Functional model
print("Starting Listing 7.16")
inputs = keras.Input(shape=(64,))
outputs = layers.Dense(64, activation="sigmoid")(inputs)
binary_classifier = keras.Model(inputs=inputs, outputs=outputs)
binary_classifier.summary()

class MyModel(keras.Model):

    def __init__(self, num_classes=2):
        super().__init__()
        self.dense = layers.Dense(64, activation="relu")
        self.classifier = binary_classifier

    def call(self, inputs):
        features = self.dense(inputs)
        return self.classifier(features)

print("Starting MyModel()")
model = MyModel()
model.build(input_shape=(64,1))

model.summary()

