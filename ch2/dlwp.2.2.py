# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Loading the MNIST dataset in Keras
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images,test_labels) = mnist.load_data()

print(train_images.shape)
print(len(train_labels))

print(test_images.shape)
print(len(test_labels))

# 2.2.1 Scalars (rank-0 tensors)
import numpy as np
x = np.array(12)
print(x)
print(f"shape: {x.shape}")
print(f"ndim: {x.ndim}")

# 2.2.2 Vectors (rank-1 tensors)
import numpy as np
x = np.array([12, 3, 6, 14, 7])
print(x)
print(f"shape: {x.shape}")
print(f"ndim: {x.ndim}")

# 2.2.3 Matrices (rank-2 tensors)
import numpy as np
x = np.array([[12, 3, 6, 14, 7],
            [13, 4, "Bob", 14, 7],
            [14, 5, 8, 14, 7]])
print(x)
print(f"dtype: {x.dtype}")
print(f"shape: {x.shape}")
print(f"ndim: {x.ndim}")

# 2.2.4 Rank-2 and higher-rank tensors
import numpy as np
x = np.array([[[12, 3, 6, 14, 7],
            [13, 4, 7, 14, 7],
            [14, 5, 8, 14, 7]],
            [[12, 3, 6, 14, 7],
            [13, 4, 7, 14, 7],
            [14, 5, 8, 14, 7]],
            [[12, 3, 6, 14, 7],
            [13, 4, 7, 14, 7],
            [14, 5, 8, 14, 7]],
            [[12, 3, 6, 14, 7],
            [13, 4, 7, 14, 7],
            [14, 5, 8, 14, 7]]])
print(x)
print(f"dtype: {x.dtype}")
print(f"shape: {x.shape}")
print(f"ndim: {x.ndim}")

# Listing 2.8 Dispaying the fourth digit
import matplotlib.pyplot as plt
digit = train_images[4]
plt.imshow(digit)
plt.show()

# 2.2.6 Manipulating tensors in numpy
my_slice = train_images[10:100]
print(f"shape: {my_slice.shape}")

my_slice = train_images[10:100, :, :]
print(f"shape: {my_slice.shape}")

my_slice = train_images[10:100, 0:27, 0:20]
print(f"shape: {my_slice.shape}")

my_slice = train_images[10:100, 0:28, 0:28]
print(f"shape: {my_slice.shape}")

my_slice = train_images[0, 0:28, 0:28]
print(f"shape: {my_slice.shape}")
print(my_slice)

my_slice = train_images[0, 7:-7, 7:-7]
print(f"shape: {my_slice.shape}")
print(my_slice)

# 2.2.8 Real-world examples of data tensors
# Timeseries Rank-3
import numpy as np
x = np.array([[["sample1-timestep1-f1", "sample1-timestep1-f2", "sample1-timestep1-f3", 14, 7],
            ["sample1-timestep2", 4, 7, 14, 7],
            ["sample1-timestep3", 5, 8, 14, 7]],
            [["sample2-timestep1-f1", "sample2-timestep1-f2", "sample2-timestep1-f3", 14, 7],
            ["sample2-timestep2", 4, 7, 14, 7],
            ["sample2-timestep3", 5, 8, 14, 7]],
            [["sample3-timestep1-f1", "sample1-timestep3-f2", "sample3-timestep1-f3", 14, 7],
            ["sample3-timestep2", 4, 7, 14, 7],
            ["sample3-timestep3", 5, 8, 14, 7]],
            [["sample4-timestep1-f1", "sample4-timestep1-f2", "sample4-timestep1-f3", 14, 7],
            ["sample4-timestep2", 4, 7, 14, 7],
            ["sample4-timestep3", 5, 8, 14, 7]]])
print(x)
print(f"dtype: {x.dtype}")
print(f"shape: {x.shape}")
print(f"ndim: {x.ndim}")

