import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define initial GDP and parameters
initial_gdp = 21000  # Initial GDP in billion USD
long_term_growth = 0.00  # Long-term expected GDP growth rate (2%)
reversion_speed = 1 # Speed at which GDP growth reverts to the mean
volatility = 0.05  # Standard deviation of shocks to GDP growth (5%)
years = 30  # Number of years for the simulation
num_simulations = 30   # Number of Monte Carlo simulations

# Set random seed for reproducibility
np.random.seed(42)

# Monte Carlo Simulation with Mean Reversion
gdp_paths = []

for _ in range(num_simulations):
    gdp = [initial_gdp]
    growth_rate = long_term_growth  # Start with the long-term growth rate
    
    for year in range(1, years + 1):
        # Simulate random shock for the year
        random_shock = np.random.normal(0, volatility)
        
        # Update growth rate with mean reversion
        growth_rate += reversion_speed * (long_term_growth - growth_rate) + random_shock
        
        # Update GDP based on the new growth rate
        gdp.append(gdp[-1] * (1 + growth_rate))
    
    gdp_paths.append(gdp)

# Convert results to a DataFrame
gdp_df = pd.DataFrame(gdp_paths).T
gdp_df.index.name = "Year"
gdp_df.columns = [f"Simulation {i+1}" for i in range(num_simulations)]

# Display some statistics of the final GDP
final_gdp = gdp_df.iloc[-1]
print(f"Mean Final GDP: {final_gdp.mean():,.2f} growth rate")
print(f"5th Percentile Final GDP growth rate: {np.percentile(final_gdp, 5):,.2f} growth rate")
print(f"95th Percentile Final GDP growth rate: {np.percentile(final_gdp, 95):,.2f} growth rate")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(gdp_df, color='lightgray', linewidth=0.5)
plt.plot(gdp_df.mean(axis=1), color='blue', linewidth=2, label='Mean GDP growth rate')
plt.title('Monte Carlo Simulation of GDP growth rate with Mean Reversion')
plt.xlabel('Year')
plt.ylabel('GDP growth rate')
plt.legend()
plt.grid(True)
plt.show()
