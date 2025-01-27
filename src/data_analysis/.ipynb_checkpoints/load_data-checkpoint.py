import pandas as pd

def load_macro_economic_data():
    """
    Loads the macro_economic.csv file into a pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    file_path = "/Users/nielsvanwinden/Projects/Projects/Inholland/Scenario_Generator/data/macro_economic.csv"
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path, delimiter=';')
        print(f"Data loaded successfully from {file_path}.")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    df = load_macro_economic_data()
    if df is not None:
        print(df.head())  # Display the first few rows of the DataFrame