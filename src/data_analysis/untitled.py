import pandas as pd
import matplotlib.pyplot as plt

def analyze_lagged_and_serial_correlation(df, column, max_lag):
    """
    Analyzes lagged correlations and serial intercorrelations (autocorrelations)
    of a time series in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the time series data.
        column (str): The column name of the time series to analyze.
        max_lag (int): The maximum number of lags to compute correlations for.
    
    Returns:
        pd.DataFrame: A DataFrame containing lag values, lagged correlations, 
                      and serial intercorrelations.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    results = []

    for lag in range(1, max_lag + 1):
        # Create lagged values
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)

        # Compute lagged correlation
        lagged_corr = df[[column, f'{column}_lag_{lag}']].corr().iloc[0, 1]

        # Compute serial intercorrelation (autocorrelation)
        autocorr = df[column].autocorr(lag=lag)

        results.append((lag, lagged_corr, autocorr))

    # Create a DataFrame for the results
    result_df = pd.DataFrame(results, columns=['Lag', 'Lagged Correlation', 'Autocorrelation'])

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(result_df['Lag'], result_df['Lagged Correlation'], marker='o', label='Lagged Correlation')
    plt.plot(result_df['Lag'], result_df['Autocorrelation'], marker='x', label='Autocorrelation', linestyle='--')
    plt.title(f'Lagged and Serial Correlation for {column}')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid()
    plt.show()

    # Drop temporary lagged columns
    lag_cols = [col for col in df.columns if col.startswith(f'{column}_lag_')]
    df.drop(columns=lag_cols, inplace=True)

    return result_df

# Example usage
if __name__ == "__main__":
    # Example DataFrame
    data = {'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
    df = pd.DataFrame(data)

    # Analyze correlations with lags up to 5
    result = analyze_lagged_and_serial_correlation(df, column='value', max_lag=5)

    # Display the results
    print(result)


import seaborn as sns
import matplotlib.pyplot as plt

def create_violin_plot(df, column, group_column):
    """
    Creates a violin plot for a time series in the given DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the time series data.
        column (str): The column name for the values to plot.
        group_column (str): The column name to group the data by (e.g., time periods).

    Returns:
        None
    """
    if column not in df.columns or group_column not in df.columns:
        raise ValueError(f"Ensure both '{column}' and '{group_column}' exist in the DataFrame.")

    # Create the violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x=group_column, y=column, scale="width", inner="quartile")
    plt.title(f'Violin Plot of {column} Grouped by {group_column}')
    plt.xlabel(group_column)
    plt.ylabel(column)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example pivot DataFrame
    data = {
        'Year': [2020, 2020, 2020, 2021, 2021, 2021],
        'Quarter': ['Q1', 'Q2', 'Q3', 'Q1', 'Q2', 'Q3'],
        'Value': [10, 20, 15, 25, 30, 20]
    }
    pivot_df = pd.DataFrame(data)

    # Create violin plot grouped by 'Quarter'
    create_violin_plot(pivot_df, column='Value', group_column='Quarter')