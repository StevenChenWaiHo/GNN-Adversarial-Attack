class CIC_IDS_2017_Config:

    DATASET_NAME = "CIC_IDS_2017"

    SOURCE_IP_COL_NAME = "Source IP"
    DESTINATION_IP_COL_NAME = "Destination IP"
    SOURCE_PORT_COL_NAME = "Source Port"
    DESTINATION_PORT_COL_NAME = "Destination Port"

    ATTACK_CLASS_COL_NAME = "Label"

    BENIGN_CLASS_NAME = "BENIGN"

    TIME_COL_NAMES = ["Timestamp"]

    INDEX_COL_NAME = "Flow ID"

    SOURCE_FILE_ID_COL_NAME = "source_file_id"

    col_names = [
        "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", "Flow Duration", 
        "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets", 
        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", 
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", 
        "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", 
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", 
        "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", 
        "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", 
        "Bwd Packets/s", "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", 
        "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", 
        "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", 
        "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length", 
        "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", 
        "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", 
        "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", 
        "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min", 
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label", "source_file_id"
    ]

    CHOSEN_COLS = [
        'Source Port', 'Destination Port', 'Bwd Packet Length Min', 'Subflow Fwd Packets',
        'Total Length of Fwd Packets', 'Fwd Packet Length Mean', 'Total Length of Fwd Packets',
        'Fwd Packet Length Std', 'Fwd IAT Min', 'Flow IAT Min', 'Flow IAT Mean', 'Bwd Packet Length Std',
        'Subflow Fwd Bytes', 'Flow Duration', 'Flow IAT Std', 'Active Min','Active Mean', 'Bwd IAT Mean',
        'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'ACK Flag Count','Fwd PSH Flags','SYN Flag Count',
        'Flow Packets/s', 'PSH Flag Count', 'Average Packet Size', 'Protocol'
    ]

    DROP_COLS = set(col_names) - set(CHOSEN_COLS) - set(TIME_COL_NAMES)
    DROP_COLS = []

    COLS_TO_NORM = [
        # 'Source Port', 'Destination Port', 
        'Bwd Packet Length Min', 'Subflow Fwd Packets',
        'Total Length of Fwd Packets', 'Fwd Packet Length Mean', 'Total Length of Fwd Packets',
        'Fwd Packet Length Std', 'Fwd IAT Min', 'Flow IAT Min', 'Flow IAT Mean', 'Bwd Packet Length Std',
        'Subflow Fwd Bytes', 'Flow Duration', 'Flow IAT Std', 'Active Min','Active Mean', 'Bwd IAT Mean',
        'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'ACK Flag Count','Fwd PSH Flags','SYN Flag Count',
        'Flow Packets/s', 'PSH Flag Count', 'Average Packet Size'
    ]

    CATEGORICAL_COLS = [
        "Protocol", 
    ]

    assert (set(COLS_TO_NORM) | set(CATEGORICAL_COLS)).issubset(set(col_names) - set(DROP_COLS)), f"Some columns are not in the right place. Please check the column names in {DATASET_NAME}_config.py ."
