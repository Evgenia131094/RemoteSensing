import os
import pandas as pd

from sklearn import metrics
import seaborn as sns

from config.config_classes import ClusterizationConfig
from Clusterization.constants import CLUSTERING_ALGORITHMS
from Utils.utils import get_config_data, save_chart


class Clusterization:
    def __init__(self, dataset: pd.DataFrame, clusterization_config: ClusterizationConfig):
        self.dataset = dataset
        self.clusterization_config = clusterization_config

    def start_training(self, get_info: bool = True):
        for clustering_algorithm in self.clusterization_config.algorithms:
            labels = self.fit(clustering_algorithm=CLUSTERING_ALGORITHMS[clustering_algorithm])
            if get_info:
                self.get_scatterplot_charts(labels=labels)

    def fit(self, clustering_algorithm, n_clusters: int = 3, random_state: int = 1):
        model = clustering_algorithm(n_clusters=n_clusters, random_state=random_state).fit(self.dataset)
        labels = model.labels_
        return labels

    def get_scatterplot_charts(self, labels):
        for column_1 in self.dataset.columns:
            for column_2 in self.dataset.columns:
                params = {"x": column_1, "y": column_2,
                          "hue": 'Clusters',
                          "data": self.dataset + labels,
                          "palette": 'viridis'}
                save_chart(method=sns.scatterplot,
                           params=params,
                           save_path=os.path.join(self.clusterization_config.info_path,
                                                  f'scatterplot_{column_1}_{column_2}.png'))


def start_clasterization(cfg_path: str, get_info: bool):
    clasterization_config = ClusterizationConfig(**get_config_data(cfg_path))
    multy_spectral_data = pd.read_csv(clasterization_config.data_annotation_path)
    clusterization = Clusterization(multy_spectral_data, clasterization_config)
    clusterization.start_training()

