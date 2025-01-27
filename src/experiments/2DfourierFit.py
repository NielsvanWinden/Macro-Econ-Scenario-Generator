import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Generate sample points with noise
def f(x):
    return np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)

# Number of sample points
num_points = 100
x = np.linspace(0, 1, num_points)
y_true = f(x)

# Add Gaussian noise
noise = np.random.normal(0, 0.2, num_points)
y_noisy = y_true + noise

# Period of the function
T = 1

# Function to compute the Fourier coefficients
def compute_fourier_coefficients(f, T, N):
    a0 = (2 / T) * quad(lambda x: f(x), 0, T)[0]
    an = lambda n: (2 / T) * quad(lambda x: f(x) * np.cos(2 * np.pi * n * x / T), 0, T)[0]
    bn = lambda n: (2 / T) * quad(lambda x: f(x) * np.sin(2 * np.pi * n * x / T), 0, T)[0]
    a = [an(n) for n in range(1, N + 1)]
    b = [bn(n) for n in range(1, N + 1)]
    return a0, a, b

# Number of Fourier terms
N = 10
a0, a, b = compute_fourier_coefficients(f, T, N)

# Function to construct the Fourier series
def fourier_series(x, a0, a, b, T):
    result = a0 / 2
    for n in range(1, len(a) + 1):
        result += a[n - 1] * np.cos(2 * np.pi * n * x / T) + b[n - 1] * np.sin(2 * np.pi * n * x / T)
    return result

# Construct the Fourier series approximation
y_fourier = fourier_series(x, a0, a, b, T)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y_true, label='Original function', color='blue')
plt.scatter(x, y_noisy, label='Noisy data', color='green', s=10)
plt.plot(x, y_fourier, label='Fourier series approximation', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Fourier Series Approximation with Noisy Data')
plt.grid(True)
plt.show()
