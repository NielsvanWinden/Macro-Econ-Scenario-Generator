import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM
import matplotlib.pyplot as plt

# Step 1: Generate Test Data
# Set a seed for reproducibility
np.random.seed(42)

# Create a date range for yearly data
dates = pd.date_range(start='1995-01-01', periods=29, freq='YE')

# Generate test data for 3 variables: Unemployment Rate, GDP Growth, HPI Growth
# Define the mean values for the variables
mean = [0.41, 0.2, 0.041]

# Define the standard deviations for the variables
std_devs = [0.001, 0.005, 0.001]

# Define the correlation coefficients
correlations = [-0.001, -0.0005, 0.0002]
unemployment_rate_gdp_correlation = -0.0001
unemployment_rate_hpi_correlation = -0.0005
gdp_hpi_correlation = 0.0002

# Create the covariance matrix
covariance_matrix = np.diag(std_devs)
covariance_matrix[0, 1] = unemployment_rate_gdp_correlation
covariance_matrix[1, 0] = unemployment_rate_gdp_correlation
covariance_matrix[0, 2] = unemployment_rate_hpi_correlation
covariance_matrix[2, 0] = unemployment_rate_hpi_correlation
covariance_matrix[1, 2] = gdp_hpi_correlation
covariance_matrix[2, 1] = gdp_hpi_correlation

# Generate samples from the multivariate normal distribution
samples = np.random.multivariate_normal(mean, covariance_matrix, size=len(dates))

# Extract the variables from the samples
unemployment_rate = samples[:, 0]
gdp_growth = samples[:, 1]
hpi_growth = samples[:, 2]

# Combine the data into a DataFrame
data = pd.DataFrame({
    'unemployment_rate': unemployment_rate,
    'gdp_growth': gdp_growth,
    'hpi_growth': hpi_growth
}, index=dates)

# Step 2: Fit a Vector Error Correction Model (VECM)
# Determine the number of cointegrating relationships
# For this example, we'll assume 1 cointegrating relationship
k_ar_diff = 5  # Number of lagged differences in VECM (equivalent to lag order - 1 in VAR)
model = VECM(data, k_ar_diff=k_ar_diff, deterministic="ci")
results = model.fit()

# Step 3: Display model summary
print(results.summary())

# Step 4: Forecasting
# Forecasting the next 30 years
forecast = results.predict(steps=30)

# Create a DataFrame for the forecasted values
forecast_index = pd.date_range(start='2024-01-01', periods=30, freq='YE')
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=data.columns)

# Add the last row of the data DataFrame to the forecast_df DataFrame
last_row = pd.DataFrame(data.iloc[-1, :].values.reshape(1, -1), index=[pd.to_datetime('2023-12-31')], columns=data.columns)
forecast_df = pd.concat([last_row, forecast_df], ignore_index=False)

# Step 5: Plot the forecasted values
plt.figure(figsize=(10, 6))
for column in data.columns:
    plt.plot(data.index, data[column], label=f'Historical {column}')
    plt.plot(forecast_df.index, forecast_df[column], linestyle='--', label=f'Forecasted {column}')

plt.title('VECM Model: Forecast of Unemployment Rate, GDP Growth, and HPI Growth')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()