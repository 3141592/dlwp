# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 3.5.1
# Listing 3.1 All-ones or all-zeros tensors
print("Listing 3.1 All-ones or all-zeros tensors")
import tensorflow as tf

x = tf.ones(shape=(2, 1))
print(x)

x = tf.zeros(shape=(2, 1))
print(x)

#
# Listing 3.2 Random tensors
print("Listing 3.2 Random tensors")
import tensorflow as tf

x = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
print(x)

x = tf.random.uniform(shape=(3, 1), minval=0., maxval=1.)
print(x)

#
# Listing 3.5 Creating a TensorFlow variable
print("Listing 3.5 Creating a TensorFlow variable")
import tensorflow as tf

v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
print(v)

#
# Listing 3.6 Assigning a value to a TensorFlow variable
print("Listing 3.6 Assigning a value to a TensorFlow variable")
v.assign(tf.ones((3, 1)))
print(v)

#
# Listing 3.7 Assigning a value to a subset of a TensorFlow variable
print("Listing 3.7 Assigning a value to a subset of a TensorFlow variable")
v[0, 0].assign(3.)
print(v)

#
# Listing 3.8 Using assign_add
print("Listing 3.8 Using assign_add")
v.assign_add(tf.ones((3, 1)))
print(v)

#
# Listing 3.9 A few basic math operations
print("Listing 3.9 A few basic math operations")
a = tf.ones((2,2))
b = tf.square(a)
c = tf.sqrt(a)
d = b + c
e = tf.matmul(a, b)
e *= d

print(a)
print(b)
print(c)
print(d)
print(e)

