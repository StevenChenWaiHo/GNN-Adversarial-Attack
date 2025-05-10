import pandas as pd
import argparse
from UNSW_NB15_config import UNSW_NB15_Config

def downsample_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Downsample the data by 10%
    normal_traffic_df = df[df[UNSW_NB15_Config.ATTACK_CLASS_COL_NAME] == UNSW_NB15_Config.BENIGN_CLASS_NAME]
    downsampled_normal_df = normal_traffic_df.sample(frac=0.1, random_state=42)
    downsampled_df = pd.concat([downsampled_normal_df, df[df[UNSW_NB15_Config.ATTACK_CLASS_COL_NAME] != UNSW_NB15_Config.BENIGN_CLASS_NAME]])
    
    # Sort the downsampled data by the specified time column
    downsampled_df = downsampled_df.sort_values(by=UNSW_NB15_Config.TIME_COL_NAMES)
    
    # Save the downsampled and sorted data to a new CSV file
    downsampled_df.to_csv(output_file, index=False)
    print(f"Downsampled CSV saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample a CSV file and sort by a time column.")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    args = parser.parse_args()

    # Generate output file name with '_downsampled' suffix
    output_file = args.input_file.replace(".csv", "_downsampled.csv")
    
    downsample_csv(args.input_file, output_file)