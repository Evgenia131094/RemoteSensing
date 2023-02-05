import pandas as pd
from dataclasses import dataclass


@dataclass
class MultySpectralDataset:
    def __init__(self, multy_spectral_data: pd.DataFrame, forbidden_features: list, target: str):
        self.multy_spectral_data = multy_spectral_data.drop(forbidden_features, axis=1)
        self.target_data = multy_spectral_data[target]
        self.columns = self.multy_spectral_data.columns
