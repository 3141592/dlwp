# 10.2 A temperature forecasting example
# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Understanding timeseries_dataset_from_array()
print("Understanding timeseries_dataset_from_array()")
from tensorflow import keras
import numpy as np

raw_data = np.arange(200)
print("raw_data: ", raw_data)
num_train_samples = raw_data.size
print("num_train_samples: ", num_train_samples)

sampling_rate = 5
sequence_length = 10
delay = sampling_rate * (sequence_length + 5)
#delay = 20
batch_size = 20

print("raw_data[:-delay]", raw_data[:-delay])
print("raw_data[5:]", raw_data[5:])
print("delay: ", delay)
print("[raw_data[:-delay][-1]]: ", [raw_data[:-delay][-1]])

train_dataset = keras.utils.timeseries_dataset_from_array(
        raw_data[:-delay],
        targets=[raw_data[:-delay][-1]],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size)

print("train_dataset:", train_dataset)

#
# Listing 10.8 Inspecting the output of one of our datasets
print("Listing 10.8 Inspecting the output of one of our datasets")
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
for samples, targets in train_dataset:
    print("sample size: ", samples.size)
    print("samples.shape: ", samples.shape)
    for i in range(samples.shape[0]):
        print([int(x) for x in samples[i]], int(targets[i]))


