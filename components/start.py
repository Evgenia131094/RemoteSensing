import os

from Preprocessing.preprocessing import start_preprocessing
from Training.Classification.classification import start_classification
from Training.Clusterization.clusterization import start_clusterization

from components.config.config_classes import StartConfig
from components.Utils.utils import get_config_data

CFG_PATH = os.environ["CFG_PATH"]
CFG_START_NAME = os.environ["CFG_START_NAME"]

MODES = {
    "preprocessing": start_preprocessing,
    "classification": start_classification,
    "clusterization": start_clusterization
}


def start_action(mode):
    print(f"Starting {mode}...")
    mode_cfg_path = os.path.join(CFG_PATH, f"{mode}_config.yaml")
    return MODES[mode](cfg_path=mode_cfg_path, get_info=True)


if __name__ == "__main__":
    start_config = StartConfig(**get_config_data(CFG_START_NAME))
    start_action(start_config.mode)
