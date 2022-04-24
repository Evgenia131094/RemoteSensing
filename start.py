import os

from Utils.utils import get_config_data
from config.config_classes import StartConfig

from Preprocessing.preprocessing import start_preprocessing
from Clusterization.clusterization import start_clasterization
from Classification.classification import start_classification


if __name__ == "__main__":
    modes = {"preprocessing": start_preprocessing,
             "classification": start_classification,
             "clasterisation": start_clasterization}
    start_cfg_path = os.path.join(os.environ["CFG_PATH"], os.environ["CFG_START_NAME"])
    mode = StartConfig(**get_config_data(start_cfg_path)).mode
    print(f"Starting {mode}...")
    mode_cfg_path = os.path.join(os.environ["CFG_PATH"], f"{mode}_config.yaml")
    modes[mode](cfg_path=mode_cfg_path, get_info=True)

