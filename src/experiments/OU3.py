import numpy as np
import pandas as pd

def simulate_second_order_ou(a1, a2, sigma, dt, n_samples, initial_conditions=(0, 0)):
    """
    Simulate a second-order OU process using the explicit solution.

    Parameters:
        a1 (float): Damping coefficient.
        a2 (float): Stiffness coefficient.
        sigma (float): Noise intensity.
        dt (float): Time step between samples.
        n_samples (int): Number of samples to generate.
        initial_conditions (tuple): Initial values for the process (X_0, dX_0).

    Returns:
        np.ndarray: Simulated data for the second-order OU process.
    """
    # Ensure stability of the system
    if a1 <= 0 or a2 <= 0:
        raise ValueError("Parameters a1 and a2 must be positive for stability.")

    # Compute natural frequency and damping ratio
    omega_0 = np.sqrt(a2)  # Natural frequency
    gamma = a1 / 2         # Damping factor
    zeta = gamma / omega_0  # Damping ratio

    # Time array
    times = np.arange(0, n_samples * dt, dt)

    # Initialize solution arrays
    X = np.zeros(n_samples)
    dX = np.zeros(n_samples)

    # Set initial conditions
    X[0] = initial_conditions[0]
    dX[0] = initial_conditions[1]

    # Generate Wiener increments
    dW = np.random.normal(0, np.sqrt(dt), size=n_samples)

    # Explicit solution based on damping regime
    if zeta < 1:  # Underdamped
        omega_d = omega_0 * np.sqrt(1 - zeta**2)  # Damped natural frequency
        for i in range(1, n_samples):
            t = times[i]
            exp_term = np.exp(-gamma * t)
            cos_term = np.cos(omega_d * t)
            sin_term = np.sin(omega_d * t)

            X[i] = (
                exp_term * (X[0] * cos_term + (dX[0] + gamma * X[0]) * sin_term / omega_d)
                + sigma * np.sum(dW[:i]) * dt  # Noise term
            )

    elif zeta == 1:  # Critically damped
        for i in range(1, n_samples):
            t = times[i]
            exp_term = np.exp(-gamma * t)
            X[i] = (
                exp_term * (X[0] + (dX[0] + gamma * X[0]) * t)
                + sigma * np.sum(dW[:i]) * dt  # Noise term
            )

    else:  # Overdamped
        r1 = -gamma + np.sqrt(gamma**2 - omega_0**2)
        r2 = -gamma - np.sqrt(gamma**2 - omega_0**2)
        for i in range(1, n_samples):
            t = times[i]
            exp_r1 = np.exp(r1 * t)
            exp_r2 = np.exp(r2 * t)

            X[i] = (
                X[0] * (r2 * exp_r1 - r1 * exp_r2) / (r2 - r1)
                + (dX[0] + gamma * X[0]) * (exp_r1 - exp_r2) / (r1 - r2)
                + sigma * np.sum(dW[:i]) * dt  # Noise term
            )

    return X


# Example Usage
if __name__ == "__main__":
    # Parameters
    a1 = 0.5  # Damping coefficient
    a2 = 0.2  # Stiffness coefficient
    sigma = 1.0  # Noise intensity
    dt = 0.1  # Time step
    n_samples = 100  # Number of samples

    # Initial conditions
    initial_conditions = (0.4, 0.0)  # Initial position and velocity

    # Simulate process
    simulated_data = simulate_second_order_ou(a1, a2, sigma, dt, n_samples, initial_conditions)

    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'time': np.arange(0, n_samples * dt, dt),
        'value': simulated_data
    })

    print(df.head())

    # Plot the simulated process
    import matplotlib.pyplot as plt
    plt.plot(df['time'], df['value'], label='Second-Order OU Process')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Simulated Second-Order OU Process')
    plt.show()