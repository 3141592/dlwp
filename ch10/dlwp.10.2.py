# 10.2 A temperature forecasting example
# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
# Listing 10.1 Inspecting the data of the Jena weather dataset
print("Listing 10.1 Inspecting the data of the Jena weather dataset")
import os
fname = os.path.join("/root/src/jena_climate_2009_2016.csv")

with open(fname) as f:
    data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
print(header)
print(len(lines))

#
# Listing 10.2 Parsing the data
print("Listing 10.2 Parsing the data")
import numpy as np
temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    temperature[i] = values[1]
    raw_data[i, :] = values[:]

#
# Listing 10.3 Plotting the temperature timeseries
print("Listing 10.3 Plotting the temperature time series")
from matplotlib import pyplot as plt
plt.plot(range(len(temperature)), temperature)
plt.show()

#
# Listing 10.4 Plotting the first 10 days of the temperature timeseries
print("Listing 10.4 Plotting the first 10 days of the temperature timeseries")
plt.plot(range(1440), temperature[:1440])
plt.show()


