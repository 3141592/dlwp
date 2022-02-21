# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Importing the library
import tensorflow as tf

def get_gradient(x):
    # Using GradientTape
    with tf.GradientTape() as gfg:
        # Starting the recording x
          gfg.watch(x)
          y = x * x * x + 4
              
    # Computing gradient
    res = gfg.gradient(y, x) 
                
    # Printing result
    print("res: ", res)

get_gradient(tf.constant(0.0))
get_gradient(tf.constant(1.0))
get_gradient(tf.constant(2.0))
get_gradient(tf.constant(3.0))
get_gradient(tf.constant(4.0))
get_gradient(tf.constant(5.0))
get_gradient(tf.constant(6.0))

