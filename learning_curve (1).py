import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given sample sizes and test accuracy values
sample_size = np.array([16, 32, 64, 128, 256, 300, 350, 380, 400])
test_accuracy = np.array([0.58, 0.62, 0.8, 0.85, 0.89, 0.95, 0.96, 0.97, 0.98])

# Define an inverse power law function to fit the data
def inverse_power_law(x, a, b, c):
    return a - b * (x ** -c)

# Fit the learning curve using curve_fit
params, covariance = curve_fit(inverse_power_law, sample_size, test_accuracy, maxfev=10000)

# Extract parameters from the fit
a, b, c = params

# Generate smooth sample size values for plotting the fitted curve
sample_size_smooth = np.linspace(min(sample_size), max(sample_size), 500)
fitted_accuracy = inverse_power_law(sample_size_smooth, a, b, c)

# Plot the original data points and the fitted learning curve
plt.figure(figsize=(10, 6))
plt.scatter(sample_size, test_accuracy, color='blue', label='Observed Accuracy', marker='o')
plt.plot(sample_size_smooth, fitted_accuracy, color='red', linestyle='--', label='Fitted Learning Curve')

# Set plot labels and title
plt.xlabel('Sample Size')
plt.ylabel('Test Accuracy')
plt.title('Learning Curve with Inverse Power Law Fit')
plt.legend()
plt.grid(True)
plt.show()
