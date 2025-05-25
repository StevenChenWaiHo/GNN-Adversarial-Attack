import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from BoT_IoT_config import BoT_IoT_Config
from sklearn.preprocessing import StandardScaler

# Paths
input_dir = './Raw'
output_all_raw_path = './All/all_raw.csv' 

CATEGORICAL_COLS = BoT_IoT_Config.CATEGORICAL_COLS
COLS_TO_NORM = BoT_IoT_Config.COLS_TO_NORM
TIME_COL_NAMES = BoT_IoT_Config.TIME_COL_NAMES

SOURCE_IP_COL_NAME = BoT_IoT_Config.SOURCE_IP_COL_NAME
DESTINATION_IP_COL_NAME = BoT_IoT_Config.DESTINATION_IP_COL_NAME

SOURCE_PORT_COL_NAME = BoT_IoT_Config.SOURCE_PORT_COL_NAME
DESTINATION_PORT_COL_NAME = BoT_IoT_Config.DESTINATION_PORT_COL_NAME

ATTACK_CLASS_COL_NAME = BoT_IoT_Config.ATTACK_CLASS_COL_NAME
BENIGN_CLASS_NAME = BoT_IoT_Config.BENIGN_CLASS_NAME

def parse_port(p):
    try:
        if isinstance(p, str) and p.lower().startswith('0x'):
            return int(p, 16)
        return int(p)
    except:
        return -1  # or np.nan if you prefer

# Combine all CSV files in the input directory
all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
df_list = []
for i, file in enumerate(all_files):
    try:
        df = pd.read_csv(file, header=0)
        df.columns = df.columns.str.strip()

        assert SOURCE_IP_COL_NAME in df.columns, f"{SOURCE_IP_COL_NAME} not found in {file}"
        assert DESTINATION_IP_COL_NAME in df.columns, f"{DESTINATION_IP_COL_NAME} not found in {file}"
        assert SOURCE_PORT_COL_NAME in df.columns, f"{SOURCE_PORT_COL_NAME} not found in {file}"
        assert DESTINATION_PORT_COL_NAME in df.columns, f"{DESTINATION_PORT_COL_NAME} not found in {file}"

        df[SOURCE_PORT_COL_NAME] = df[SOURCE_PORT_COL_NAME].astype(str).apply(parse_port)
        df[DESTINATION_PORT_COL_NAME] = df[DESTINATION_PORT_COL_NAME].astype(str).apply(parse_port)
        
        df_list.append(df)
    except UnicodeDecodeError as e:
        print(f"Error in file: {file}")
        print(f"Error: {e}")

df_full = pd.concat(df_list)

# ==== Essential Preprocessing START ====

# Strip whitespaces from all string columns
for col in df_full.select_dtypes(include='object').columns:
    df_full[col] = df_full[col].str.strip()

# Fill NaNs
df_full.fillna(0, inplace=True)

sort_by_cols = TIME_COL_NAMES

df_full = df_full.sort_values(by=sort_by_cols)
sorted_correctly = df_full[sort_by_cols].astype(str).apply(tuple, axis=1).is_monotonic_increasing
assert sorted_correctly, "DataFrame is not sorted by TIME_COL_NAMES"

# ==== Essential Preprocessing END ====

df_full.to_csv(output_all_raw_path, index=False, header=True)
print(f"Raw CSV saved to {output_all_raw_path}")