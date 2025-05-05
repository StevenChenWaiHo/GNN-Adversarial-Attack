import os
import sys
import pandas as pd
import pickle

def csv_to_dataframe(input_csv_path):
    # Create the output binary file path
    output_dir = os.path.dirname(input_csv_path)
    output_file = os.path.join(output_dir, os.path.basename(input_csv_path) + ".pkl")

    # Check if the pickle file exists
    if os.path.isfile(output_file):
        try:
            with open(output_file, 'rb') as f:
                df = pickle.load(f)
            print(f"DataFrame loaded from existing pickle file: '{output_file}'")
            return df
        except Exception as e:
            print(f"Error loading the pickle file: {e}")
            sys.exit(1)

    # If pickle file doesn't exist, read the CSV into a DataFrame
    if not os.path.isfile(input_csv_path):
        print(f"Error: The file '{input_csv_path}' does not exist.")
        sys.exit(1)

    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    # Save the DataFrame as a pickle file
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(df, f)
        print(f"DataFrame successfully saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving the DataFrame: {e}")
        sys.exit(1)

    return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_dataframe.py <input_csv_path>")
        sys.exit(1)

    input_csv_path = sys.argv[1]
    df = csv_to_dataframe(input_csv_path)
    print(df.head())
    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns.tolist())