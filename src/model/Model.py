import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# Get the current script's directory
src_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(src_dir)
from model import *

class Model:
    def __init__(self, mu: np.ndarray, X_0: np.ndarray, n: int, T: int, dt: float, A: np.ndarray, Sigma: np.ndarray):
        """Stochastic model constants."""
        self.n = n # Number of components
        self.T = T  # Number of time steps
        self.dt = dt # Time step size
        self.A = A # Drift term
        self.Sigma = Sigma # Volatility term
        self.X_0 = X_0 # Current macro realisation

        """Initalize externally provided baseline mu_t (time-varying) and timeseries X."""
        self.mu = mu

    # For testing purposes only
    def __init__(self):
        # Initialize parameters
        self.n = 2 # Number of components
        self.T = 30  # Number of time steps
        self.dt = 1  # Time step size
        self.A = np.array(
            [[1, 0],  # Drift matrix
            [0, 1]]
        )  
        self.Sigma = np.array(
            [[0.007239375880210823, 0],  # Noise covariance matrix        
            [0, 00.016912]]
        )

        # Initialize data
        self.X_0 = np.array([7, 3.5])
        self.mu = np.zeros((self.T, self.n))
        self.time = np.linspace(0, 4 * np.pi, self.T)  # Define time for sinusoidal trend over 2 full periods
        self.mu[:, 0] = 0.052333  # Sinusoidal trend for the first component
        self.mu[:, 1] = 0.1 * np.sin(self.time) + 0.041 # Sinusoidal trend for the second component (phase-shifted)

    def generate_projection(self) -> Tuple[np.ndarray, np.ndarray]:
        return generate_projection(mu = self.mu, X_0 = self.X_0, n = self.n, T = self.T, dt = self.dt, A = self.A, Sigma = self.Sigma)

if __name__ == '__main__':
    model = Model()
    mu, X = model.generate_projection()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(X[:, 0], label="X1 (Simulated)")
    plt.plot(mu[:, 0], linestyle='--', label="mu1 (Baseline)")
    plt.plot(X[:, 1], label="X2 (Simulated)")
    plt.plot(mu[:, 1], linestyle='--', label="mu2 (Baseline)")
    plt.legend()
    plt.title("SDE Simulation with Externally Provided Baseline Mu")
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.show()