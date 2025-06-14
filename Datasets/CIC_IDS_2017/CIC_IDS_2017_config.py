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

    COL_NAMES = [
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

    DROP_COLS = [
        "Flow Bytes/s", "Idle Min", "Bwd Packet Length Mean", "Init_Win_bytes_backward", 
        "Bwd IAT Max", "Bwd PSH Flags", "Packet Length Mean", "Packet Length Variance", 
        "Fwd IAT Std", "Down/Up Ratio", "Bwd IAT Std", 
        "Bwd Packets/s", "Idle Mean", "RST Flag Count", "Fwd Avg Bytes/Bulk", 
        "Fwd IAT Max", "Fwd Packets/s", "Avg Fwd Segment Size", "Fwd Avg Bulk Rate", 
        "Bwd Avg Packets/Bulk", "Bwd URG Flags", "Bwd Header Length", "Bwd Avg Bulk Rate", 
        "Avg Bwd Segment Size", "Min Packet Length", "URG Flag Count", "Fwd Avg Packets/Bulk", 
        "Total Backward Packets", "Max Packet Length", "Fwd Packet Length Min", "CWE Flag Count", 
        "FIN Flag Count", "act_data_pkt_fwd", "Subflow Bwd Packets", 
        "Bwd Avg Bytes/Bulk", "Flow IAT Max", "Fwd Header Length", 
        "Active Std", "min_seg_size_forward", "Fwd IAT Total", 
        "Bwd Packet Length Max", "ECE Flag Count", "Fwd Packet Length Max", 
        "Total Length of Bwd Packets", "Total Fwd Packets", "Idle Max", 
        "Active Max", "Idle Std", "Fwd URG Flags", "Fwd IAT Mean", "Bwd IAT Total", 
        "Bwd IAT Min", "Packet Length Std",
    ]

    assert (set(COLS_TO_NORM) | set(CATEGORICAL_COLS)).issubset(set(COL_NAMES) - set(DROP_COLS)), f"Some columns are not in the right place. Please check the column names in {DATASET_NAME}_config.py ."
