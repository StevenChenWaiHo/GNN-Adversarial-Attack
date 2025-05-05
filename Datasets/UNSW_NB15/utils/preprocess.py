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
    
    return dataframe