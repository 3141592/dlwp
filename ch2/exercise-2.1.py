import numpy as np
import matplotlib.pyplot as plt

print("1. Prepare training data")
x = np.array([1, 2, 3, 4, 5])
targets = np.array([2, 4, 6, 8, 10])

print("1a. Normalize x")
x = x / 5
print(f"x: {x}")
print(f"targets: {targets}")

print("")
print("2. Initialize model parameters")
W = np.random.randn() # single weight for simplicity
b = np.random.randn() # Bias
print(f"W: {W}")
print(f"b: {b}")

print("")
print("3. Define the prediction function")
y_pred = W*x + b
print(f"y_pred: {y_pred}")

print("")
print("4. Calculate the loss")
loss_mse = np.mean((y_pred - targets) ** 2)
print(f"loss_mse: {loss_mse}")

loss_m_1_8e = np.mean((y_pred - targets) ** 1.8)
print(f"loss_m_1_8e: {loss_m_1_8e}")

print("")
print("5. Compute gradients")

print("")
print("6. Update parameters")

print("")
print("7. Iterate")

print("")
print("8. Test the model")

