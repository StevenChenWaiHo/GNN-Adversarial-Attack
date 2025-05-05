import os
import pandas as pd
from sklearn.model_selection import train_test_split
from UNSW_NB15_config import UNSW_NB15_Config

# Paths
input_dir = './All'
output_train_path = './Train/train.csv'
output_eval_path = './Eval/eval.csv'

# Combine all CSV files in the input directory
all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
df_list = [pd.read_csv(file, header=None, names=UNSW_NB15_Config.COL_NAMES) for file in all_files]  # Use the header list from config
df_full = pd.concat(df_list, ignore_index=True)

# Set 'attack_cat' column to 'Normal' if 'Label' column is 0
df_full.loc[df_full[UNSW_NB15_Config.IS_ATTACK_COL_NAME] == 0, UNSW_NB15_Config.ATTACK_CLASS_COL_NAME] = UNSW_NB15_Config.BENIGN_CLASS_NAME

# Split the dataset
train_df, test_df = train_test_split(
    df_full,
    test_size=82332,
    stratify=df_full[UNSW_NB15_Config.ATTACK_CLASS_COL_NAME],
    random_state=42
)

# Save the split datasets to CSV
os.makedirs('./Train', exist_ok=True)
os.makedirs('./Eval', exist_ok=True)
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