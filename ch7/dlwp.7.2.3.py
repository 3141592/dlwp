# 7.2.3 Subclassing the Model class
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

# Listing 7.14 A simple subclassed model
class CustomerTicketModel(keras.Model):
    # Call the super() constructor!
    super().__init__()
    self.concat_layer = layers.Concatenate()
    self.mixing_layer = layers.Dense(64, activation="relu")
    self.priority_scorer = layers.Dense(1, activation="sigmoid")
    self.department_classifier = layers.Dense(num_departments, activation="sofmax")


