import yaml
import matplotlib.pyplot as plt


def get_config_data(path_to_config: str):
    with open(path_to_config) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_chart(method, params: dict, save_path: str):
    figure, _ = plt.subplots(figsize=(10, 8))
    method(**params)
    figure.savefig(save_path, dpi=800)
