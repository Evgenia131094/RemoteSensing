import os

import pandas as pd
import seaborn as sns
from Utils.utils import save_chart


class Info:

    def __init__(self, data, info_path):
        self.data = data
        self.info_path = info_path

    def get_dataset_info_in_charts(self):
        print("Get correlation heatmap")
        self.get_correlation_heatmap()
        print("Get histograms...")
        self.get_histograms()
        print("Get pair plots...")
        self.get_pair_plots()

    def get_correlation_heatmap(self, method: str = 'pearson'):  # 'penguins'
        df = self.data.multy_spectral_data.copy()
        df["target"] = self.data.target_data.astype(int)
        corr = df.dropna().corr(method=method)
        save_chart(method=sns.heatmap,
                   save_path=os.path.join(self.info_path,
                                          f'correlations_{method}.png'),
                   data=corr,
                   annot=True)

    def get_histograms(self):
        df = self.data.multy_spectral_data.copy()
        df["target"] = self.data.target_data.astype(int)
        for column in df:
            save_chart(method=sns.kdeplot,
                       save_path=os.path.join(self.info_path,
                                              f'histogram_{column}.png'),
                       x=self.data.multy_spectral_data[column],
                       shade=True)

    def get_pair_plots(self, hue: str = "Green"):
        df = self.data.multy_spectral_data.copy()
        df["target"] = self.data.target_data.astype(int)

        save_chart(method=sns.pairplot,
                   save_path=os.path.join(self.info_path, f'pairplot.png'),
                   data=df,
                   hue=hue)
