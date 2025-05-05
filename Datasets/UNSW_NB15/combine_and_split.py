import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from UNSW_NB15_config import UNSW_NB15_Config
from utils.preprocess import preprocess
from sklearn.preprocessing import StandardScaler

# Paths
input_dir = './All'
output_all_raw_path = './All/all_raw.csv' 
output_all_path = './All/all_preprocessed.csv' 
output_train_path = './Train/train_preprocessed.csv'
output_eval_path = './Eval/eval_preprocessed.csv'

# Combine all CSV files in the input directory
all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
df_list = [pd.read_csv(file, header=None, names=UNSW_NB15_Config.COL_NAMES) for file in all_files]  # Use the header list from config
df_full = pd.concat(df_list, ignore_index=True)

# Strip whitespaces from all string columns
df_full = df_full.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Set 'attack_cat' column to 'Normal' if 'Label' column is 0
df_full.loc[df_full[UNSW_NB15_Config.IS_ATTACK_COL_NAME] == 0, UNSW_NB15_Config.ATTACK_CLASS_COL_NAME] = UNSW_NB15_Config.BENIGN_CLASS_NAME

# Fill NaNs (optional: set to 0 or use imputation)
df_full.fillna(0, inplace=True)

df_full = df_full.sort_values(by=UNSW_NB15_Config.TIME_COL_NAMES)

df_full.to_csv(output_all_raw_path, index=False, header=True)

df_full = preprocess(df_full)

# Categorical columns to one-hot encode
df_full = pd.get_dummies(df_full, columns=UNSW_NB15_Config.CATEGORICAL_COLS, drop_first=True)

print(f"=====Columns=====")
print(df_full.columns.tolist())
print('No. of Columns: ', len(df_full.columns.tolist()))

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

check_numeric_issues(df_full, UNSW_NB15_Config.COLS_TO_NORM)

# Split the dataset
train_df, test_df = train_test_split(
    df_full,
    test_size=82332,
    stratify=df_full[UNSW_NB15_Config.ATTACK_CLASS_COL_NAME],
    random_state=42
)

scaler = StandardScaler()
cols_to_norm = UNSW_NB15_Config.COLS_TO_NORM
print(train_df[cols_to_norm].describe()) # Check if there's any too large value

train_df = train_df.sort_values(by=UNSW_NB15_Config.TIME_COL_NAMES)
test_df = test_df.sort_values(by=UNSW_NB15_Config.TIME_COL_NAMES)

train_df[cols_to_norm] = scaler.fit_transform(train_df[cols_to_norm])
test_df[cols_to_norm] = scaler.transform(test_df[cols_to_norm])

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')

# Save the split datasets to CSV
os.makedirs('./Train', exist_ok=True)
os.makedirs('./Eval', exist_ok=True)
df_full.to_csv(output_all_path, index=False, header=True)
train_df.to_csv(output_train_path, index=False, header=True)
test_df.to_csv(output_eval_path, index=False, header=True)

print(f"=====Train dataset=====")
print(f"Train dataset shape: {train_df.shape}")
print(train_df[UNSW_NB15_Config.ATTACK_CLASS_COL_NAME].value_counts())
print(train_df[UNSW_NB15_Config.IS_ATTACK_COL_NAME].value_counts())


print(f"=====Eval dataset=====")
print(f"Eval dataset shape: {test_df.shape}")
print(test_df[UNSW_NB15_Config.ATTACK_CLASS_COL_NAME].value_counts())
print(test_df[UNSW_NB15_Config.IS_ATTACK_COL_NAME].value_counts())

print(f"Train dataset saved to {output_train_path}")
print(f"Eval dataset saved to {output_eval_path}")