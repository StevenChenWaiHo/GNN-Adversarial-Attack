class BoT_IoT_Config:

    DATASET_NAME = "BoT_IoT"

    SOURCE_IP_COL_NAME = 'saddr'
    DESTINATION_IP_COL_NAME = 'daddr'
    SOURCE_PORT_COL_NAME = 'sport'
    DESTINATION_PORT_COL_NAME = 'dport'

    ATTACK_CLASS_COL_NAME = 'label'

    BENIGN_CLASS_NAME = "Normal"

    TIME_COL_NAMES = ['stime']

    INDEX_COL_NAME = "pkSeqID"

    # SOURCE_FILE_ID_COL_NAME = "source_file_id"

    col_names = [
        "pkSeqID","stime","flgs","flgs_number","proto","proto_number","saddr","sport","daddr","dport","pkts","bytes","state","state_number","ltime","seq","dur","mean","stddev","sum","min","max","spkts","dpkts","sbytes","dbytes","rate","srate","drate","TnBPSrcIP","TnBPDstIP","TnP_PSrcIP","TnP_PDstIP","TnP_PerProto","TnP_Per_Dport","AR_P_Proto_P_SrcIP","AR_P_Proto_P_DstIP","N_IN_Conn_P_DstIP","N_IN_Conn_P_SrcIP","AR_P_Proto_P_Sport","AR_P_Proto_P_Dport","Pkts_P_State_P_Protocol_P_DestIP","Pkts_P_State_P_Protocol_P_SrcIP","attack","category","subcategory"
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

    CATEGORICAL_COLS = ['flgs_number','state_number', 'proto_number']

    DROP_COLS = ['subcategory','pkSeqID','stime','flgs','attack','state','proto','seq']

    assert (set(COLS_TO_NORM) | set(CATEGORICAL_COLS)).issubset(set(col_names) - set(DROP_COLS)), f"Some columns are not in the right place. Please check the column names in {DATASET_NAME}_config.py ."
