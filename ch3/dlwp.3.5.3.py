# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
# Listing 3.10 Using the GradientTape
print("Listing 3.10 Using the GradientTape")
import tensorflow as tf

input_var = tf.Variable(3.)

#
#input_var = tf.compat.v1.get_variable(initial_value=3.)
with tf.GradientTape() as tape:
    result = tf.square(input_var)
gradient = tape.gradient(result, input_var)
print(gradient)

#
# Listing 3.11 Using GradientTape with constant tensor inputs
print("Listing 3.11 Using GradientTape with constant tensor inputs")
input_const = tf.constant(3.)
with tf.GradientTape() as tape:
    tape.watch(input_const)
    result = tf.square(input_const)
gradient = tape.gradient(result, input_const)

#
# Listing 3.12 Using nested gradient tapes to compute second-order gradients
print("Listing 3.12 Using nested gradient tapes to compute second-order gradients")
time = tf.Variable(0.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time ** 2
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)
print(acceleration)


