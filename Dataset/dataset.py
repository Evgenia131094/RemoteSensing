import os
import pandas as pd
import seaborn as sns

from Utils.utils import save_chart


class MultySpectralDataset:
    def __init__(self, multy_spectral_data: pd.DataFrame, info_path: str):
        self.multy_spectral_data_without_target = multy_spectral_data.drop(['image', 'class'], axis=1)
        self.target_data = multy_spectral_data["class"]
        self.info_path = info_path
        self.columns = self.multy_spectral_data_without_target.columns

    def get_dataset_info_in_charts(self):
        print("Get correlation heatmap")
        self.get_correlation_heatmap()
        print("Get histograms...")
        self.get_histograms()
        print("Get pair plots...")
        self.get_pair_plots()

    def get_correlation_heatmap(self, method: str = 'pearson'):  #'penguins'
        corr = self.multy_spectral_data_without_target.corr(method=method)
        params = {"data": corr, "annot": True}
        save_chart(method=sns.heatmap,
                   params=params,
                   save_path=os.path.join(self.info_path, f'correlations_{method}.png'))

    def get_histograms(self):
        for column in self.multy_spectral_data_without_target.columns:
            params = {"x": self.multy_spectral_data_without_target[column], "shade": True}
            save_chart(method=sns.kdeplot,
                       params=params,
                       save_path=os.path.join(self.info_path, f'histogram_{column}.png'))

    def get_pair_plots(self, hue: str = "Green"):
        params = {"data": self.multy_spectral_data_without_target, "hue": hue}
        save_chart(method=sns.pairplot,
                   params=params,
                   save_path=os.path.join(self.info_path, f'pairplot.png'))
