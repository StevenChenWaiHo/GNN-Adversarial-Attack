import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from CIC_IDS_2017_config import CIC_IDS_2017_Config
from sklearn.preprocessing import StandardScaler
import numpy as np

# Paths
input_dir = './Raw/'
output_all_raw_path = './All/all_raw.csv' 
output_all_downsampled_path = './All/all_downsampled.csv' 
output_train_path = './Train/train_preprocessed.csv'
output_eval_path = './Eval/eval_preprocessed.csv'

SOURCE_IP_COL_NAME = CIC_IDS_2017_Config.SOURCE_IP_COL_NAME
DESTINATION_IP_COL_NAME = CIC_IDS_2017_Config.DESTINATION_IP_COL_NAME
SOURCE_PORT_COL_NAME = CIC_IDS_2017_Config.SOURCE_PORT_COL_NAME
DESTINATION_PORT_COL_NAME = CIC_IDS_2017_Config.DESTINATION_PORT_COL_NAME

ATTACK_CLASS_COL_NAME = CIC_IDS_2017_Config.ATTACK_CLASS_COL_NAME
INDEX_COL_NAME = CIC_IDS_2017_Config.INDEX_COL_NAME

# Combine all CSV files in the input directory
all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]

df_list = []
for i, file in enumerate(all_files):
    try:
        df = pd.read_csv(file, header=0, encoding='cp1252')

        assert SOURCE_IP_COL_NAME in df.columns, f"{SOURCE_IP_COL_NAME} not found in {file}"
        assert DESTINATION_IP_COL_NAME in df.columns, f"{DESTINATION_IP_COL_NAME} not found in {file}"

        df.columns = df.columns.str.strip()
        if SOURCE_IP_COL_NAME in df.columns:
            df[SOURCE_IP_COL_NAME] = df[SOURCE_IP_COL_NAME].astype(str).apply(lambda x: f"{x}_{i}")
        if DESTINATION_IP_COL_NAME in df.columns:
            df[DESTINATION_IP_COL_NAME] = df[DESTINATION_IP_COL_NAME].astype(str).apply(lambda x: f"{x}_{i}")
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

df_full.fillna(0, inplace=True)

# # Replace – with -
df_full[ATTACK_CLASS_COL_NAME] = (
    df_full[ATTACK_CLASS_COL_NAME]
    .str.replace('ï¿½', '-', regex=False)
    .str.replace('–', '-', regex=False)  # en dash
    .str.replace('—', '-', regex=False)  # em dash
    .str.strip()
)

# ==== Essential Preprocessing END ====

print(df_full[ATTACK_CLASS_COL_NAME].value_counts())

if 'Web Attack - Brute Force' not in df_full[ATTACK_CLASS_COL_NAME].unique():
    raise ValueError("Web Attack - Brute Force not found in the dataset")

df_full.to_csv(output_all_raw_path, index=False, header=True)

# Downsample the data by 90%
normal_traffic_df = df_full[df_full[CIC_IDS_2017_Config.ATTACK_CLASS_COL_NAME] == CIC_IDS_2017_Config.BENIGN_CLASS_NAME]
downsampled_normal_df = normal_traffic_df.sample(frac=0.1, random_state=42)
downsampled_df = pd.concat([downsampled_normal_df, df_full[df_full[CIC_IDS_2017_Config.ATTACK_CLASS_COL_NAME] != CIC_IDS_2017_Config.BENIGN_CLASS_NAME]])

downsampled_df.to_csv(output_all_downsampled_path, index=False, header=True)

# ==== Optional Preprocessing START ====
downsampled_preprocessed = downsampled_df.drop(columns=CIC_IDS_2017_Config.DROP_COLS)

# 2. Convert columns to appropriate data types
downsampled_preprocessed[SOURCE_IP_COL_NAME] = downsampled_preprocessed[SOURCE_IP_COL_NAME].apply(str)
downsampled_preprocessed[SOURCE_PORT_COL_NAME] = downsampled_preprocessed[SOURCE_PORT_COL_NAME].apply(str)
downsampled_preprocessed[DESTINATION_IP_COL_NAME] = downsampled_preprocessed[DESTINATION_IP_COL_NAME].apply(str)
downsampled_preprocessed[DESTINATION_PORT_COL_NAME] = downsampled_preprocessed[DESTINATION_PORT_COL_NAME].apply(str)

print(downsampled_preprocessed.head)

# 3. Handle missing values
downsampled_preprocessed = downsampled_preprocessed.reset_index()
downsampled_preprocessed.replace([np.inf, -np.inf], np.nan,inplace = True)
downsampled_preprocessed.fillna(0,inplace = True)
downsampled_preprocessed.drop(columns=[INDEX_COL_NAME],inplace=True)

# 4. Encode categorical columns
downsampled_preprocessed = pd.get_dummies(downsampled_preprocessed, columns=CIC_IDS_2017_Config.CATEGORICAL_COLS)
bool_cols = downsampled_preprocessed.select_dtypes(include='bool').columns
downsampled_preprocessed[bool_cols] = downsampled_preprocessed[bool_cols].astype(int)


print(f"=====Columns=====")
print(downsampled_preprocessed.columns.tolist())
print('No. of Columns: ', len(downsampled_preprocessed.columns.tolist()))

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

check_numeric_issues(downsampled_preprocessed, CIC_IDS_2017_Config.COLS_TO_NORM)

# ==== Optional Preprocessing END ====

print("downsampled: ", downsampled_preprocessed.head(5))
# Save the downsampled and sorted data to a new CSV file

# Split the dataset
train_df, test_df = train_test_split(
    downsampled_preprocessed,
    test_size=0.15,
    stratify=downsampled_preprocessed[CIC_IDS_2017_Config.ATTACK_CLASS_COL_NAME],
    random_state=42
)

scaler = StandardScaler()
cols_to_norm = CIC_IDS_2017_Config.COLS_TO_NORM
print(train_df[cols_to_norm].describe()) # Check if there's any too large value

train_df[cols_to_norm] = scaler.fit_transform(train_df[cols_to_norm])
test_df[cols_to_norm] = scaler.transform(test_df[cols_to_norm])

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')

# Save the split datasets to CSV
os.makedirs('./Train', exist_ok=True)
os.makedirs('./Eval', exist_ok=True)

train_df.to_csv(output_train_path, index=False, header=True)
test_df.to_csv(output_eval_path, index=False, header=True)

print(f"=====Train dataset=====")
print(f"Train dataset shape: {train_df.shape}")
print(train_df[CIC_IDS_2017_Config.ATTACK_CLASS_COL_NAME].value_counts())


print(f"=====Eval dataset=====")
print(f"Eval dataset shape: {test_df.shape}")
print(test_df[CIC_IDS_2017_Config.ATTACK_CLASS_COL_NAME].value_counts())

print(f"Train dataset saved to {output_train_path}")
print(f"Eval dataset saved to {output_eval_path}")