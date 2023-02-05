import os
import yaml
import matplotlib.pyplot as plt
from pathlib import Path


def get_config_data(path_to_config: str):
    with open(path_to_config) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def create_save_dir(dir_path: str, method: str):
    save_dir = os.path.join(dir_path, method)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir


def save_chart(method, save_path: str, **kwargs):
    figure, _ = plt.subplots(figsize=(10, 8))
    method(**kwargs)
    plt.savefig(save_path, dpi=800)
