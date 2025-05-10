import os

class UNSW_NB15_Config:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Datasets/"))
    DATASET_DIR = os.path.join(BASE_DIR, "UNSW_NB15/")

    DATASET_NAME = "UNSW_NB15"

    ALL_PATH = DATASET_DIR + "All/all_preprocessed.csv"
    TRAIN_PATH = DATASET_DIR + "Train/train_preprocessed.csv"
    TEST_PATH = DATASET_DIR + "Eval/eval_preprocessed.csv"

    DATASET_GRAPH_DIR = "./logs/UNSW_NB15/graph/"
    TRAIN_GRAPH_PATH = DATASET_GRAPH_DIR + "Train/train_graph.pt"
    TEST_GRAPH_PATH = DATASET_GRAPH_DIR + "Eval/eval_graph.pt"

    SOURCE_NODE = "srcip"
    DESTINATION_NODE = "dstip"

    SOURCE_PORT_COL_NAME = "sport"
    DESTINATION_PORT_COL_NAME = "dsport"

    ATTACK_CLASS_COL_NAME = "attack_cat"
    IS_ATTACK_COL_NAME = "label"

    BENIGN_CLASS_NAME = "Normal"

    TIME_COL_NAMES = ["Stime", "Ltime"]

