from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

def generate_projection(mu: np.ndarray, X_0: np.ndarray,n: int, T: int, dt: float, A: np.ndarray, Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    # Parameter Validattion
    assert isinstance(mu, np.ndarray), "mu must be of type np.ndarray"
    assert isinstance(n, int), "n must be of type int"
    assert isinstance(T, int), "T must be of type int"
    assert isinstance(dt, float) or isinstance(dt, int), "dt must be of type float or int"
    assert isinstance(A, np.ndarray), "A must be of type np.ndarray"
    assert isinstance(Sigma, np.ndarray), "Sigma must be of type np.ndarray"
    
    # Initialize the process
    X = np.zeros((T, n))

    X[0] =  X_0
    noise = np.random.multivariate_normal(mean=np.zeros(n), cov=Sigma, size=T)
    
    # Simulate the process
    for t in range(T - 1):
        trend_adjustment = (mu[t + 1] - mu[t]) / dt
        drift = A @ (mu[t] - X[t]) * dt + trend_adjustment * dt
        diffusion = noise[t] * np.sqrt(dt)
        X[t + 1] = X[t] + drift + diffusion
    
    return mu, X


if __name__ == "__main__":
    # Set seed
    np.random.seed(42) 
    
    # Parameters
    n = 2 # Number of components
    T = 30  # Number of time steps
    dt = 1  # Time step size
    A = np.array(
        [[0.5, 0],  # Drift matrix
        [0, 0.5]]
    )  
    Sigma = np.array(
        [[0.92, 0.05],  # Noise covariance matrix        
        [0.05, 0.6]]
    )  
    
    # Externally provided baseline mu_t (time-varying)
    X_0 = np.array([7, 3.5])
    mu = np.zeros((T, n))
    time = np.linspace(0, 4 * np.pi, T)  # Define time for sinusoidal trend over 2 full periods
    mu[:, 0] = 1 * time + 7  # Sinusoidal trend for the first component
    mu[:, 1] = -0.5 * time + 3.5 # Sinusoidal trend for the second component (phase-shifted)
    
    # # Externally provided baseline mu_t (time-varying)
    # X_0 = np.array([5, 5])
    # mu = np.zeros((T, n))
    # time = np.linspace(0, 4 * np.pi, T)  # Define time for sinusoidal trend over 2 full periods
    # mu[:, 0] = 3 * np.sin(time) + 5  # Sinusoidal trend for the first component
    # mu[:, 1] = 2 * np.sin(time + np.pi / 2) + 3  # Sinusoidal trend for the second component (phase-shifted)
    
    mu, X = generate_projection(mu=mu,X_0=X_0, n=n, T=T, dt=dt, A=A, Sigma=Sigma)
        