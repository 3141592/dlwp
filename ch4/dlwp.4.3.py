# 4.3 Predicting house prices: A regression example
# Listing 4.23 Loading the Boston housing dataset
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import boston_housing

import numpy as np

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(f"train_data.shape: {train_data.shape}")
print(f"test_data.shape: {test_data.shape}")
#print(f"train_targets: {train_targets}")

# 4.3.2 Preparing the data
# Listing 4.24 Normalizing data
mean = train_data.mean(axis=0)
print(f"train_data[0]: {train_data[0]}")
train_data -= mean
print(f"train_data[0]: {train_data[0]}")
std = train_data.std(axis=0)
train_data /= std
print(f"train_data[0]: {train_data[0]}")
test_data -= mean
test_data /= std

# 4.3.3 Building your model
# Listing 4.25 Model definition
def build_model():
    model = keras.Sequential([
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop",
            loss="mse",
            metrics=["mae"])
    return model

# 4.34 Validating your approach using K-fold validation
# Listing 4.26 K-fold validation
"""
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)
    model = build_model()
    model.fit(partial_train_data,
            partial_train_targets,
            epochs=num_epochs,
            batch_size=16,
            verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    print(f"all_scores: {all_scores}")
    print(f"np.mean(all_scores): {np.mean(all_scores)}")
"""

# Listing 4.27 Saving the validation logs at each fold
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)
    model = build_model()
    history = model.fit(partial_train_data,
                partial_train_targets,
                validation_data=(val_data, val_targets),
                epochs=num_epochs,
                batch_size=16,
                verbose=0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)

# Listing 4.28 Building the history of successive mean K-fold validation scores
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# Listing 4.29 Plotting validation scores
import matplotlib.pyplot as plt
"""
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()
"""
# Listing 4.30 Plotting validation scores, excluding the first 10 data points
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
print("Uncomment next line for plot 4.30")
#plt.show()

# Listing 4.31 Training the final model
model = build_model()
model.fit(train_data,
        train_targets,
        epochs=130,
        batch_size=16,
        verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(f"test_mse_score: {test_mse_score}")
print(f"test_mae_score: {test_mae_score}")

"""
Model 2 Layers Dense 64
Run 1
test_mse_score: 14.543720245361328
test_mae_score: 2.4559051990509033

Run2
test_mse_score: 16.470335006713867
test_mae_score: 2.4394171237945557

Model 3 Layers Dense 64
Run 1
test_mse_score: 13.875872611999512
test_mae_score: 2.540989637374878

Model 2 Layers Dense 32
Run 1
test_mse_score: 17.631181716918945
test_mae_score: 2.5997469425201416

"""

# 4.3.5 Generating predictions on new data
predictions = model.predict(test_data)
print(f"predictions[0]: {predictions[0]}")


