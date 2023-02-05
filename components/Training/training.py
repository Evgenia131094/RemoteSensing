import os
import pickle
import inspect
from components.Dataset.dataset import MultySpectralDataset


class Training:
    def __init__(self, dataset: MultySpectralDataset, config):
        self.dataset = dataset
        self.config = config

    def fit(self, data, training_algorithm, **kwargs):
        model = self._init_train_algorithm(algorithm=training_algorithm, input_params=kwargs).fit(data)
        return model

    @staticmethod
    def save_model(model, save_dir):
        model_path = os.path.join(save_dir, 'model.pickle')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def _init_train_algorithm(algorithm, input_params: dict):
        model_valid_parameters = inspect.getfullargspec(algorithm).args
        input_params = {key: value for key, value in input_params.items() if key in model_valid_parameters}
        return algorithm(**input_params)
