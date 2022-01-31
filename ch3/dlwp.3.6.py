import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

# Listing 3.22 A Dense layer implemented as a Layer subclass
class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        print("In __init__")
        super().__init__()
        self.units = units
        self.activation = activation

    # weight creation takes place in the build() method
    def build(self, input_shape):
        print("In build()")
        input_dim = input_shape[-1]

        # add_weight is a shortcut method for creating weights.
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer="random_normal")
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros")

    # Define the forward pass computation in the call() method
    def call(self, inputs):
        print("In call()")
        print(inputs)
        print(self.W)
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

# Test SimpleDense class
units = 32
print(f"units= {units}")
my_dense = SimpleDense(units=units, activation=tf.nn.relu)
input_tensor = tf.ones(shape=(3, 1))
output_tensor = my_dense(input_tensor)

#print(input_tensor[0])
#print(output_tensor.shape)
print(output_tensor)

