import numpy as np
import pandas as pd


def second_order_ou_likelihood(params, data, dt):
    """
    Compute the negative log-likelihood for a second-order OU process.

    Parameters:
        params (list): [a1, a2, sigma], the parameters to estimate.
        data (np.ndarray): Time series data (observed values).
        dt (float): Time step between observations.

    Returns:
        float: Negative log-likelihood.
    """
    a1, a2, sigma = params

    # Ensure stability of the system
    if a1 <= 0 or a2 <= 0:
        return np.inf

    n = len(data)
    residuals = np.zeros(n - 2)

    # Compute residuals based on the discretized second-order OU process
    for t in range(2, n):
        # Discretized second-order OU approximation
        predicted = (
            (2 - a1 * dt) * data[t - 1]
            - (1 - a2 * dt**2) * data[t - 2]
        )
        residuals[t - 2] = data[t] - predicted

    # Variance of the noise term
    residual_var = sigma**2 * dt
    if residual_var <= 0:
        return np.inf

    # Negative log-likelihood calculation
    neg_log_likelihood = 0.5 * np.sum(
        np.log(2 * np.pi * residual_var) + (residuals**2) / residual_var
    )
    return neg_log_likelihood

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
    n_samples = 100

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