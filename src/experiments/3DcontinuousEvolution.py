import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters for each dimension
a = [1.0, 0.5, 1.5]  # Coefficients of t
b = [2.0, 1.0, 3.0]  # Constant terms
mu = [0, 0, 0]  # Drift coefficients
sigma = [1.0, 0.8, 1.2]  # Diffusion coefficients
t_max = 10  # Maximum time
dt = 0.00001  # Time step
n_steps = int(t_max / dt)
n_simulations = 5  # Number of stochastic simulations to visualize

# Time array
t = np.linspace(0, t_max, n_steps)

# Simulate and plot multiple stochastic projections in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for _ in range(n_simulations):
    # Simulate Wiener processes for each dimension
    W1 = np.random.normal(0, np.sqrt(dt), size=n_steps).cumsum()
    W2 = np.random.normal(0, np.sqrt(dt), size=n_steps).cumsum()
    W3 = np.random.normal(0, np.sqrt(dt), size=n_steps).cumsum()

    # Calculate X(t) for each dimension
    X1 = mu[0] * t + sigma[0] * W1
    X2 = mu[1] * t + sigma[1] * W2
    X3 = mu[2] * t + sigma[2] * W3

    # Calculate Y(t) for each dimension
    Y1 = (a[0] + mu[0]) * t + b[0] + sigma[0] * W1
    Y2 = (a[1] + mu[1]) * t + b[1] + sigma[1] * W2
    Y3 = (a[2] + mu[2]) * t + b[2] + sigma[2] * W3

    # Plot stochastic projection in 3D
    ax.plot(Y1, Y2, Y3, linestyle='dashed', alpha=0.7)

# Plot the linear projection (without stochastic component)
Y1_linear = a[0] * t + b[0]
Y2_linear = a[1] * t + b[1]
Y3_linear = a[2] * t + b[2]
ax.plot(Y1_linear, Y2_linear, Y3_linear, color='black', linewidth=2, label='Linear Projection')

ax.set_title('3D Stochastic Projections')
ax.set_xlabel('Y1(t)')
ax.set_ylabel('Y2(t)')
ax.set_zlabel('Y3(t)')
ax.legend()
plt.show()
