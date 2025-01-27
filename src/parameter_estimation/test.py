import numpy as np
import pandas as pd
from estimation import second_order_ou_likelihood

def generate_second_order_ou_data(a1, a2, sigma, dt, n_samples, initial_conditions=(0, 0)):
    """
    Generate synthetic data for a second-order OU process using a discretized approximation.

    Parameters:
        a1 (float): Damping coefficient.
        a2 (float): Stiffness coefficient.
        sigma (float): Noise intensity.
        dt (float): Time step between samples.
        n_samples (int): Number of samples to generate.
        initial_conditions (tuple): Initial values for the process (X_0, X_1).

    Returns:
        np.ndarray: Simulated data for the second-order OU process.
    """
    X = np.zeros(n_samples)
    noise = np.random.normal(0, sigma * np.sqrt(dt), size=n_samples)

    # Set initial conditions
    X[0] = initial_conditions[0]
    if n_samples > 1:
        X[1] = initial_conditions[1]

    # Simulate the second-order OU process
    for t in range(2, n_samples):
        X[t] = (2 - a1 * dt) * X[t - 1] - (1 - a2 * dt**2) * X[t - 2] + noise[t]

    return X


def test_second_order_ou_likelihood():
    """
    Test the second_order_ou_likelihood function.
    """
    # True parameters for the second-order OU process
    true_a1 = 0.55
    true_a2 = 0.25
    true_sigma = 1.05
    dt = 0.1  # Time step (days)
    n_samples = 11200

    # Generate synthetic data
    np.random.seed(42)  # For reproducibility
    synthetic_data = generate_second_order_ou_data(true_a1, true_a2, true_sigma, dt, n_samples)

    # Define a function to compute the likelihood with given parameters
    def likelihood_for_params(params):
        return second_order_ou_likelihood(params, synthetic_data, dt)

    # Test: Likelihood at true parameters should be minimal
    true_params = [true_a1, true_a2, true_sigma]
    likelihood_true = likelihood_for_params(true_params)

    # Test: Perturb parameters and check if likelihood increases
    perturbed_params = [0.6, 0.3, 1.2]  # Slightly incorrect parameters
    likelihood_perturbed = likelihood_for_params(perturbed_params)

    print("Likelihood at true parameters:", likelihood_true)
    print("Likelihood at perturbed parameters:", likelihood_perturbed)

    assert likelihood_true < likelihood_perturbed, (
        "Likelihood at true parameters should be smaller than at perturbed parameters."
    )

    # Test: Minimization recovers true parameters
    from scipy.optimize import minimize

    initial_guess = [0.4, 0.3, 0.8]  # Initial guess for parameters
    result = minimize(
        second_order_ou_likelihood,
        x0=initial_guess,
        args=(synthetic_data, dt),
        bounds=[(1e-5, None), (1e-5, None), (1e-5, None)],
        method="L-BFGS-B",
    )

    estimated_params = result.x
    print("Estimated Parameters:", estimated_params)

    # Assert the estimated parameters are close to the true ones
    np.testing.assert_allclose(
        estimated_params, true_params, rtol=1e-1, atol=1e-1,
        err_msg="Estimated parameters do not match the true parameters."
    )

    print("All tests passed!")

# Run the test
if __name__ == "__main__":
    test_second_order_ou_likelihood()