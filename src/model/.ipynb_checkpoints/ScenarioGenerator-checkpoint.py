import numpy as np
import matplotlib.pyplot as plt
import sys
# Define the path to the parent directory of data_analysis
parent_dir = '/Users/nielsvanwinden/Projects/Projects/Inholland/Scenario_Generator/src'
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from generate_projection import *



class ScenarioGenerator:
    def __init__(self, mu: np.ndarray, X_0: np.ndarray,n: int, T: int, dt: float, A: np.ndarray, Sigma: np.ndarray):
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
            [[0.5, 0],  # Drift matrix
            [0, 0.5]] 
        )  
        self.Sigma = np.array(
            [[0.92, 0.05],  # Noise covariance matrix        
            [0.05, 0.6]]
        )

        # Initialize data
        self.X_0 = np.array([7, 3.5])
        self.mu = np.zeros((self.T, self.n))
        self.time = np.linspace(0, 4 * np.pi, T)  # Define time for sinusoidal trend over 2 full periods
        self.mu[:, 0] = 1 * np.sin(self.time) + 7  # Sinusoidal trend for the first component
        self.mu[:, 1] = -0.5 * np.sin(self.time) + 3.5 # Sinusoidal trend for the second component (phase-shifted)

    def generate_projection(self) -> Tuple[np.ndarray, np.ndarray] -> Tuple[np.ndarray, np.ndarray]:
        return generate_projection(mu = self.mu, X_0 = self.X_0, T = self.T, dt = self.dt, A = self.A, Sigma = self.Sigma)