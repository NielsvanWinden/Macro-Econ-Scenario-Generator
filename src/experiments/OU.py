import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Example DataFrame
df = pd.DataFrame({
    'pit_data': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'time_series_value': np.random.randn(100)
})

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

    # Unpack parameters
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


def estimate_ou_parameters(df):
    """
    Estimate second-order OU parameters for the time series data in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with 'pit_data' and 'time_series_value' columns.

    Returns:
        dict: Estimated parameters (a1, a2, sigma).
    """
    # Extract time intervals (assume homogeneously spaced data)
    df = df.sort_values(by='pit_data')
    dt = (df['pit_data'].iloc[1] - df['pit_data'].iloc[0]).total_seconds()
    dt = dt / (60 * 60 * 24)  # Convert to days

    # Extract time series data
    data = df['time_series_value'].values

    # Initial guesses for parameters [a1, a2, sigma]
    initial_guess = [0.1, 0.1, 1.0]

    # Minimize negative log-likelihood
    result = minimize(
        second_order_ou_likelihood,
        x0=initial_guess,
        args=(data, dt),
        bounds=[(1e-5, None), (1e-5, None), (1e-5, None)],
        method='L-BFGS-B',
    )

    # Extract results
    if result.success:
        a1, a2, sigma = result.x
        return {'a1': a1, 'a2': a2, 'sigma': sigma}
    else:
        raise ValueError("Parameter estimation failed: " + result.message)


# Example Usage
if __name__ == "__main__":
    # Create a dummy DataFrame with evenly spaced timestamps and sample values
    df = pd.DataFrame({
        'pit_data': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'time_series_value': np.cumsum(np.random.randn(100))  # Example process
    })

    # Estimate parameters
    params = estimate_ou_parameters(df)
    print("Estimated Parameters:", params)

    # Extract parameters
    a1 = params['a1']
    a2 = params['a2']
    sigma = params['sigma']

    # Example likelihood calculation
    dt = (df['pit_data'].iloc[1] - df['pit_data'].iloc[0]).total_seconds() / (60 * 60 * 24)  # Convert to days
    data = df['time_series_value'].values
    likelihood = second_order_ou_likelihood([a1, a2, sigma], data, dt)
    print("Negative Log-Likelihood:", likelihood)


import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Function definitions from your existing code remain the same:
# second_order_ou_likelihood(params, data, dt)
# estimate_ou_parameters(df)

def calculate_aic_bic(neg_log_likelihood, n_params, n_data):
    """
    Calculate AIC and BIC given the negative log-likelihood, number of parameters, and dataset size.

    Parameters:
        neg_log_likelihood (float): The negative log-likelihood of the model.
        n_params (int): The number of parameters in the model.
        n_data (int): The number of data points in the dataset.

    Returns:
        dict: A dictionary containing AIC and BIC values.
    """
    aic = 2 * n_params + 2 * neg_log_likelihood
    bic = n_params * np.log(n_data) + 2 * neg_log_likelihood
    return {"AIC": aic, "BIC": bic}

# Example Usage
if __name__ == "__main__":
    # Create a dummy DataFrame with evenly spaced timestamps and sample values
    df = pd.DataFrame({
        'pit_data': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'time_series_value': np.cumsum(np.random.randn(100))  # Example process
    })

    # Estimate parameters
    params = estimate_ou_parameters(df)
    print("Estimated Parameters:", params)

    # Extract parameters and compute negative log-likelihood
    a1 = params['a1']
    a2 = params['a2']
    sigma = params['sigma']
    dt = (df['pit_data'].iloc[1] - df['pit_data'].iloc[0]).total_seconds() / (60 * 60 * 24)  # Convert to days
    data = df['time_series_value'].values
    neg_log_likelihood = second_order_ou_likelihood([a1, a2, sigma], data, dt)
    print("Negative Log-Likelihood:", neg_log_likelihood)

    # Compute AIC and BIC
    n_params = 3  # Number of parameters (a1, a2, sigma)
    n_data = len(data)
    criteria = calculate_aic_bic(neg_log_likelihood, n_params, n_data)
    print("AIC:", criteria["AIC"])
    print("BIC:", criteria["BIC"])