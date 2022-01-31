# 5.2 Evaluating machine learning models
# 5.2.1 Training, validation, and test sets

from tensorflow.keras.datasets import mnist
import numpy as np

(data, train_labels), _ = mnist.load_data()
data = data.reshape((60000, 28 * 28))
#data = data.astype("float32") / 255

# Listing 5.5 Holdout validation
num_validation_sample = 10000
np.random.shuffle(data)
validation_data = data[:num_validation_sample]
training_data = data[num_validation_sample:]
print(f"len(validation_data): {len(validation_data)}")
print(f"len(training_data): {len(training_data)}")

# Listing 5.6 K-fold cross-validation
k = 3
num_validation_samples = len(data) // k
np.random.shuffle(data)
validation_scores = []
for fold in range(k):
    validation_data = data[num_validation_samples * fold:
                           num_validation_samples * (fold + 1)]
    training_data = np.concatenate((
            data[:num_validation_samples * fold],
            data[num_validation_samples * (fold + 1):]))
    print(f"Fold: {fold}")
    print(f"len(validation_data): {len(validation_data)}")
    print(f"len(training_data): {len(training_data)}")

