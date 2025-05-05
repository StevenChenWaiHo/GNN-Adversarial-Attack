class UNSW_NB15_Config:

    DATASET_NAME = "UNSW_NB15"

    SOURCE_IP_COL_NAME = "srcip"
    DESTINATION_IP_COL_NAME = "dstip"
    SOURCE_PORT_COL_NAME = "sport"
    DESTINATION_PORT_COL_NAME = "dsport"

    ATTACK_CLASS_COL_NAME = "attack_cat"
    IS_ATTACK_COL_NAME = "label"

    BENIGN_CLASS_NAME = "Normal"

    COL_NAMES = [
        "srcip",
        "sport",
        "dstip",
        "dsport",
        "proto",
        "state",
        "dur",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sloss",
        "dloss",
        "service",
        "Sload",
        "Dload",
        "Spkts",
        "Dpkts",
        "swin",
        "dwin",
        "stcpb",
        "dtcpb",
        "smeansz",
        "dmeansz",
        "trans_depth",
        "res_bdy_len",
        "Sjit",
        "Djit",
        "Stime",
        "Ltime",
        "Sintpkt",
        "Dintpkt",
        "tcprtt",
        "synack",
        "ackdat",
        "is_sm_ips_ports",
        "ct_state_ttl",
        "ct_flw_http_mthd",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_srv_src",
        "ct_srv_dst",
        "ct_dst_ltm",
        "ct_src_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        "attack_cat",
        "label",
    ]

    # DROP_COLS = ['subcategory','pkSeqID','stime','flgs','state','proto','seq']

    DROP_COLS = ["proto", "service"]

    COLS_TO_NORM = [
        # "srcip",
        # "sport",
        # "dstip",
        # "dsport",
        # "proto",
        # "state",
        "dur",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sloss",
        "dloss",
        # "service",
        "Sload",
        "Dload",
        "Spkts",
        "Dpkts",
        "swin",
        "dwin",
        "stcpb",
        "dtcpb",
        "smeansz",
        "dmeansz",
        "trans_depth",
        "res_bdy_len",
        "Sjit",
        "Djit",
        "Stime",
        "Ltime",
        "Sintpkt",
        "Dintpkt",
        "tcprtt",
        "synack",
        "ackdat",
        "is_sm_ips_ports",
        "ct_state_ttl",
        "ct_flw_http_mthd",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_srv_src",
        "ct_srv_dst",
        "ct_dst_ltm",
        "ct_src_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        # "attack_cat",
        # "label",
    ]

    CATEGORICAL_COLS = [
        "state", # Indicates to the state and its dependent protocol, e.g. ACC, CLO, CON, ECO, ECR, FIN, INT, MAS, PAR, REQ, RST, TST, TXD, URH, URN, and (-) (if not used state)
    ]
