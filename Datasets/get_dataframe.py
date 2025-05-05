import importlib
import os
import sys
import pandas as pd
import pickle

def preprocess_df(dataframe, dataset_dir, preprocess=True):
    if not preprocess:
        return dataframe
    try:
        # Make dataset_dir importable
        if dataset_dir not in sys.path:
            sys.path.insert(0, dataset_dir)
        
        # Build the full path to preprocess.py
        preprocess_script = os.path.join(dataset_dir, "preprocess.py")
        
        if not os.path.exists(preprocess_script):
            print(f"Error: '{preprocess_script}' does not exist.")
            sys.exit(1)

        # Dynamically load the module
        spec = importlib.util.spec_from_file_location("preprocess_module", preprocess_script)
        preprocess_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preprocess_module)

        # Call the preprocess function inside preprocess.py
        if not hasattr(preprocess_module, "preprocess"):
            print("Error: No 'preprocess' function found in the preprocess module.")
            sys.exit(1)

        return preprocess_module.preprocess(dataframe)

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)

def csv_to_dataframe(input_csv_path, preprocess=True):
    # Create the output binary file path
    output_dir = os.path.dirname(input_csv_path)
    output_file = os.path.join(output_dir, os.path.basename(input_csv_path) + ".pkl")

    dataset_dir = os.path.abspath(os.path.join(input_csv_path, "../../"))

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

    df = preprocess_df(df, dataset_dir, preprocess)

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
        print("Usage: python get_dataframe.py <input_csv_path> [Optional: preprocess(True/False)]")
        sys.exit(1)

    input_csv_path = sys.argv[1]
    preprocess = sys.argv[2] if len(sys.argv) == 3 else True
    df = csv_to_dataframe(input_csv_path, preprocess)
    print(df.head())
    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns.tolist())