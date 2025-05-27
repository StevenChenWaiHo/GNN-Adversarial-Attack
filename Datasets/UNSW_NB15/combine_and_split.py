import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from UNSW_NB15_config import UNSW_NB15_Config
from utils.preprocess import preprocess
from sklearn.preprocessing import StandardScaler

# Paths
input_dir = './Raw'
output_all_raw_path = './All/all_raw.csv' 
output_all_downsampled_path = './All/all_downsampled.csv' 
output_all_processed_path = './All/all_preprocessed_downsampled.csv' 
output_train_path = './Train/train_preprocessed.csv'
output_eval_path = './Eval/eval_preprocessed.csv'

COL_NAMES = UNSW_NB15_Config.COL_NAMES
CATEGORICAL_COLS = UNSW_NB15_Config.CATEGORICAL_COLS
COLS_TO_NORM = UNSW_NB15_Config.COLS_TO_NORM
TIME_COL_NAMES = UNSW_NB15_Config.TIME_COL_NAMES

SOURCE_IP_COL_NAME = UNSW_NB15_Config.SOURCE_IP_COL_NAME
DESTINATION_IP_COL_NAME = UNSW_NB15_Config.DESTINATION_IP_COL_NAME

SOURCE_PORT_COL_NAME = UNSW_NB15_Config.SOURCE_PORT_COL_NAME
DESTINATION_PORT_COL_NAME = UNSW_NB15_Config.DESTINATION_PORT_COL_NAME

ATTACK_CLASS_COL_NAME = UNSW_NB15_Config.ATTACK_CLASS_COL_NAME
BENIGN_CLASS_NAME = UNSW_NB15_Config.BENIGN_CLASS_NAME
IS_ATTACK_COL_NAME = UNSW_NB15_Config.IS_ATTACK_COL_NAME

SOURCE_FILE_ID_COL_NAME = UNSW_NB15_Config.SOURCE_FILE_ID_COL_NAME

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
        df = pd.read_csv(file, header=None, names=COL_NAMES)
        df.columns = df.columns.str.strip()

        assert SOURCE_IP_COL_NAME in df.columns, f"{SOURCE_IP_COL_NAME} not found in {file}"
        assert DESTINATION_IP_COL_NAME in df.columns, f"{DESTINATION_IP_COL_NAME} not found in {file}"
        assert SOURCE_PORT_COL_NAME in df.columns, f"{SOURCE_PORT_COL_NAME} not found in {file}"
        assert DESTINATION_PORT_COL_NAME in df.columns, f"{DESTINATION_PORT_COL_NAME} not found in {file}"

        df[SOURCE_PORT_COL_NAME] = df[SOURCE_PORT_COL_NAME].astype(str).apply(parse_port)
        df[DESTINATION_PORT_COL_NAME] = df[DESTINATION_PORT_COL_NAME].astype(str).apply(parse_port)
        
        df['source_file_id'] = i

        df_list.append(df)
    except UnicodeDecodeError as e:
        print(f"Error in file: {file}")
        print(f"Error: {e}")

df_full = pd.concat(df_list)

# ==== Essential Preprocessing START ====

# Strip whitespaces from all string columns
for col in df_full.select_dtypes(include='object').columns:
    df_full[col] = df_full[col].str.strip()

# Set 'attack_cat' column to 'Normal' if 'Label' column is 0
df_full.loc[df_full[IS_ATTACK_COL_NAME] == 0, ATTACK_CLASS_COL_NAME] = BENIGN_CLASS_NAME

# Fill NaNs
df_full.fillna(0, inplace=True)

sort_by_cols = [SOURCE_FILE_ID_COL_NAME] + TIME_COL_NAMES

df_full = df_full.sort_values(by=sort_by_cols)
sorted_correctly = df_full[sort_by_cols].astype(str).apply(tuple, axis=1).is_monotonic_increasing
assert sorted_correctly, "DataFrame is not sorted by TIME_COL_NAMES"

# Relabel misspelled classes backdoor -> backdoors
df_full[ATTACK_CLASS_COL_NAME] = df_full[ATTACK_CLASS_COL_NAME].replace({'Backdoor': 'Backdoors'})

# ==== Essential Preprocessing END ====

df_full.to_csv(output_all_raw_path, index=False, header=True)
print(f"Raw CSV saved to {output_all_raw_path}")

# Downsample the data by 10%
normal_traffic_df = df_full[df_full[ATTACK_CLASS_COL_NAME] == BENIGN_CLASS_NAME]
downsampled_normal_df = normal_traffic_df.sample(frac=0.1, random_state=42)
downsampled_df = pd.concat([downsampled_normal_df, df_full[df_full[ATTACK_CLASS_COL_NAME] != BENIGN_CLASS_NAME]])

# Sort the downsampled data by the specified time column
downsampled_df = downsampled_df.sort_values(by=sort_by_cols)

# Save the downsampled and sorted data to a new CSV file
downsampled_df.to_csv(output_all_downsampled_path, index=False)
print(f"Downsampled CSV saved to {output_all_downsampled_path}")

exit()

df_processed = preprocess(downsampled_df)

# Categorical columns to one-hot encode
df_processed = pd.get_dummies(df_processed, columns=CATEGORICAL_COLS, drop_first=True)

print(f"=====Columns=====")
print(df_processed.columns.tolist())
print('No. of Columns: ', len(df_processed.columns.tolist()))

def check_numeric_issues(df, cols_to_norm):
    for col in cols_to_norm:
        try:
            # Try to coerce to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Try to clip the column
            df[col] = df[col].clip(lower=-1e9, upper=1e9)
            
        except Exception as e:
            print(f"❌ Column '{col}' failed with error: {e}")
            print(f"  - Sample values: {df[col].dropna().unique()[:5]}")
            print(f"  - Data type: {df[col].dtype}")
            continue

    print("\n✅ All other columns processed successfully.")

check_numeric_issues(df_processed, COLS_TO_NORM)

# Split the dataset
train_df, test_df = train_test_split(
    df_processed,
    test_size=0.15,
    stratify=df_processed[ATTACK_CLASS_COL_NAME],
    random_state=42
)

scaler = StandardScaler()
cols_to_norm = COLS_TO_NORM
print(train_df[cols_to_norm].describe()) # Check if there's any too large value

train_df = train_df.sort_values(by=TIME_COL_NAMES)
test_df = test_df.sort_values(by=TIME_COL_NAMES)

train_df[cols_to_norm] = scaler.fit_transform(train_df[cols_to_norm])
test_df[cols_to_norm] = scaler.transform(test_df[cols_to_norm])

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')

# Save the split datasets to CSV
os.makedirs('./Train', exist_ok=True)
os.makedirs('./Eval', exist_ok=True)
df_processed.to_csv(output_all_processed_path, index=False, header=True)
train_df.to_csv(output_train_path, index=False, header=True)
test_df.to_csv(output_eval_path, index=False, header=True)

print(f"=====Train dataset=====")
print(f"Train dataset shape: {train_df.shape}")
print(train_df[ATTACK_CLASS_COL_NAME].value_counts())
print(train_df[IS_ATTACK_COL_NAME].value_counts())

print(f"=====Eval dataset=====")
print(f"Eval dataset shape: {test_df.shape}")
print(test_df[ATTACK_CLASS_COL_NAME].value_counts())
print(test_df[IS_ATTACK_COL_NAME].value_counts())

print(f"Train dataset saved to {output_train_path}")
print(f"Eval dataset saved to {output_eval_path}")