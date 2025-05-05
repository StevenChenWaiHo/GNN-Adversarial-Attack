from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
from UNSW_NB15_config import UNSW_NB15_Config

def preprocess(dataframe):
    """
    Preprocess the UNSW-NB15 dataset.

    Args:
        dataframe (pd.DataFrame): The input dataframe to preprocess.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """

    SOURCE_IP_COL_NAME = UNSW_NB15_Config.SOURCE_IP_COL_NAME
    DESTINATION_IP_COL_NAME = UNSW_NB15_Config.DESTINATION_IP_COL_NAME
    SOURCE_PORT_COL_NAME = UNSW_NB15_Config.SOURCE_PORT_COL_NAME
    DESTINATION_PORT_COL_NAME = UNSW_NB15_Config.DESTINATION_PORT_COL_NAME

    ATTACK_CLASS_COL_NAME = UNSW_NB15_Config.ATTACK_CLASS_COL_NAME
    IS_ATTACK_COL_NAME = UNSW_NB15_Config.IS_ATTACK_COL_NAME

    print(dataframe[ATTACK_CLASS_COL_NAME].value_counts())

    # Drop unnecessary columns
    dataframe.drop(columns=UNSW_NB15_Config.DROP_COLS,inplace=True)

    print(dataframe[IS_ATTACK_COL_NAME].value_counts())

    dataframe[SOURCE_IP_COL_NAME] = dataframe[SOURCE_IP_COL_NAME].apply(str)
    dataframe[SOURCE_PORT_COL_NAME] = dataframe[SOURCE_PORT_COL_NAME].apply(str)
    dataframe[DESTINATION_IP_COL_NAME] = dataframe[DESTINATION_IP_COL_NAME].apply(str)
    dataframe[DESTINATION_PORT_COL_NAME] = dataframe[DESTINATION_PORT_COL_NAME].apply(str)
    dataframe[SOURCE_IP_COL_NAME] = dataframe[SOURCE_IP_COL_NAME] + ':' + dataframe[SOURCE_PORT_COL_NAME]
    dataframe[DESTINATION_IP_COL_NAME] = dataframe[DESTINATION_IP_COL_NAME] + ':' + dataframe[DESTINATION_PORT_COL_NAME]
    dataframe.drop(columns=[SOURCE_PORT_COL_NAME,DESTINATION_PORT_COL_NAME],inplace=True)

    print(dataframe.head)

    dataframe = dataframe.reset_index()
    dataframe.replace([np.inf, -np.inf], np.nan,inplace = True)
    dataframe.fillna(0,inplace = True)
    dataframe.drop(columns=['index'],inplace=True)

    print(dataframe.head)

    scaler = StandardScaler()
    cols_to_norm = UNSW_NB15_Config.COLS_TO_NORM
    print(dataframe[cols_to_norm].describe()) # Check if there's any too large value

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

    check_numeric_issues(dataframe, UNSW_NB15_Config.COLS_TO_NORM)

    # Convert categorical columns to one-hot encoding
    dataframe = pd.get_dummies(dataframe, columns=UNSW_NB15_Config.CATEGORICAL_COLS, drop_first=True)

    dataframe[cols_to_norm] = scaler.fit_transform(dataframe[cols_to_norm])

    # Save the scaler to a file
    joblib.dump(scaler, 'scaler.pkl')
    return dataframe