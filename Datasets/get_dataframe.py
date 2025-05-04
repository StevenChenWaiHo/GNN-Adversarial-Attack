import os
import sys
import pandas as pd
import pickle

def csv_to_dataframe(input_csv_path):
    # Check if the input file exists
    if not os.path.isfile(input_csv_path):
        print(f"Error: The file '{input_csv_path}' does not exist.")
        sys.exit(1)

    # Read the CSV into a DataFrame
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    # Create the output binary file path
    output_dir = os.path.dirname(input_csv_path)
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_csv_path))[0] + ".pkl")

    # Save the DataFrame as a binary file
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(df, f)
        print(f"DataFrame successfully saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving the DataFrame: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_dataframe.py <input_csv_path>")
        sys.exit(1)

    input_csv_path = sys.argv[1]
    csv_to_dataframe(input_csv_path)