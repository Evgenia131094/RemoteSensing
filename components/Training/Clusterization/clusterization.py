import os
from datetime import datetime

import pandas as pd
import seaborn as sns
import umap
from tqdm import tqdm

from components.config.config_classes import TrainConfig
from components.Dataset.dataset import MultySpectralDataset
from Training.constants import CLUSTERING_ALGORITHMS
from components.Training.training import Training
from components.Utils.utils import create_save_dir, get_config_data, save_chart


class Clusterization(Training):

    def start_training(self, get_info: bool = True):
        print(f"Fitting {self.config.algorithms}")

        experiment_start_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        for clustering_algorithm in tqdm(self.config.algorithms[:]):
            model = self.fit(
                data=self.dataset.multy_spectral_data,
                training_algorithm=CLUSTERING_ALGORITHMS[clustering_algorithm],
                n_clusters=10,
                random_state=1)

            labels = model.labels_
            save_dir = create_save_dir(
                self.config.model_path,
                f"{experiment_start_time}_{clustering_algorithm}")
            self.save_model(model, save_dir)


            save_dir = create_save_dir(
                self.config.info_path,
                f"{experiment_start_time}_{clustering_algorithm}")
            self.get_umap_scatter(labels=labels, save_dir=save_dir)
            self.get_scatterplot(labels=labels, save_dir=save_dir)

    def get_umap_scatter(self,
                         save_dir: str,
                         labels,
                         palette: str = 'viridis'):
        embedding = umap.UMAP(n_neighbors=50, min_dist=0).fit_transform(
            self.dataset.multy_spectral_data[::])
        save_chart(
            method=sns.scatterplot,
            save_path=os.path.join(save_dir, f'scatterplot_0_50.png'),
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue='Clusters',
            data=self.dataset.multy_spectral_data[::].assign(Clusters=labels),
            palette=palette)

    def get_scatterplot(self, labels, save_dir: str, palette: str = 'viridis'):
        print("Start scatterplot ...")
        processed_cols = set()
        for column_1 in self.dataset.columns:
            print(f"Draw scatterplot for {column_1}")
            for column_2 in tqdm(set(self.dataset.columns) - processed_cols):
                save_chart(method=sns.scatterplot,
                           save_path=os.path.join(
                               save_dir,
                               f'scatterplot_{column_1}_{column_2}.png'),
                           x=column_1,
                           y=column_2,
                           hue='Clusters',
                           data=self.dataset.multy_spectral_data.assign(
                               Clusters=labels),
                           palette=palette)

            processed_cols.add(column_1)


def start_clusterization(cfg_path: str, get_info: bool = True):
    clasterization_config = TrainConfig(**get_config_data(cfg_path))
    multy_spectral_data = pd.read_csv(
        clasterization_config.data_annotation_path)
    multy_spectral_dataset = MultySpectralDataset(
        multy_spectral_data,
        forbidden_features=["image", "class"],
        target="class")
    clusterization = Clusterization(multy_spectral_dataset,
                                    clasterization_config)
    clusterization.start_training()
