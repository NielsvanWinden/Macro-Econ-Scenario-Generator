import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 0.5  # Coefficient of t
b = 2.0  # Constant term
mu = 0  # Drift coefficient
sigma = 1  # Diffusion coefficient
t_max = 30  # Maximum time
dt = 0.0001  # Time step
n_steps = int(t_max / dt)
n_simulations = 20  # Number of stochastic simulations to visualize

# Time array
t = np.linspace(0, t_max, n_steps)

# Linear projection without stochastic component
Y_linear = a * t + b

# Plot the linear projection
plt.figure(figsize=(10, 6))
plt.plot(t, Y_linear, label='Y(t) = at + b', color='black', linewidth=2)

# Simulate and plot multiple stochastic projections
for _ in range(n_simulations):
    # Simulate Wiener process
    W = np.random.normal(0, np.sqrt(dt), size=n_steps).cumsum()
    # Calculate X(t)
    X = mu * t + sigma * W
    # Calculate Y
    Y_stochastic = (a + mu) * t + b + sigma * W
    # Plot stochastic projection
    plt.plot(t, Y_stochastic, linestyle='dashed', alpha=0.7)

# Labeling
plt.title('Stochastic Projections')
plt.xlabel('Time t')
plt.ylabel('Value')
plt.legend(['Y(t) = at + b', 'Y(t) with stochastic component'])
plt.grid(True)

plt.show()
