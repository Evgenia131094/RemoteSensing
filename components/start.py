import os

from Preprocessing.preprocessing import start_preprocessing
from Training.Clusterization.clusterization import start_clusterization
from Training.Classification.classification import start_classification

CFG_PATH = os.environ["CFG_PATH"]

MODES = {"preprocessing": start_preprocessing,
         "classification": start_classification,
         "clusterization": start_clusterization}


def start_action(mode):
    print(f"Starting {mode}...")
    mode_cfg_path = os.path.join(CFG_PATH, f"{mode}_config.yaml")
    return MODES[mode](cfg_path=mode_cfg_path, get_info=True)

