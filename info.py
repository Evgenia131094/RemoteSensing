from Utils.utils import save_chart
import os
import seaborn as sns


class Info:
    def __init__(self, data):
        self.data = data

    def get_dataset_info_in_charts(self):
        print("Get correlation heatmap")
        self.get_correlation_heatmap()
        print("Get histograms...")
        self.get_histograms()
        print("Get pair plots...")
        self.get_pair_plots()

    def get_correlation_heatmap(self, method: str = 'pearson'):  #'penguins'
        corr = self.data.multy_spectral_data.corr(method=method)
        save_chart(method=sns.heatmap,
                   save_path=os.path.join(self.data.info_path, f'correlations_{method}.png'),
                   data=corr,
                   annot=True)

    def get_histograms(self):
        for column in self.data.multy_spectral_data.columns:
            save_chart(method=sns.kdeplot,
                       save_path=os.path.join(self.data.info_path, f'histogram_{column}.png'),
                       x=self.data.multy_spectral_data[column],
                       shade=True)

    def get_pair_plots(self, hue: str = "Green"):
        save_chart(method=sns.pairplot,
                   save_path=os.path.join(self.data.info_path, f'pairplot.png'),
                   data=self.data.multy_spectral_data,
                   hue=hue
                   )
