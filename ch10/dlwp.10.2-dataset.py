# 10.2 A temperature forecasting example
# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Understanding timeseries_dataset_from_array()
print("Understanding timeseries_dataset_from_array()")
from tensorflow import keras
import numpy as np

raw_data = np.arange(300)
print("raw_data: ", raw_data)
print("raw_data[:-3]: ", raw_data[:-3])
print("raw_data[3:]: ", raw_data[3:])

train_dataset = keras.utils.timeseries_dataset_from_array(
        data=raw_data,
        targets=raw_data[3:],
        sequence_length=5,
        batch_size=5,
        start_index=100,
        end_index=121)

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


