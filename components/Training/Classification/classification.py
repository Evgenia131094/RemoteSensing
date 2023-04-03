import os
from datetime import datetime

import pandas as pd
import shap
from sklearn.metrics import classification_report
from sklearn.model_selection import ShuffleSplit, train_test_split
from tqdm import tqdm
from Training.constants import CLASSIFICATION_ALGORITHMS, TREE_ALGORITHMS

from components.config.config_classes import TrainConfig
from components.Dataset.dataset import MultySpectralDataset
from components.Training.training import Training
from components.Utils.utils import create_save_dir, get_config_data, save_chart


class Classification(Training):

    def _fit(self, x, y, training_algorithm, **kwargs):
        model = self._init_train_algorithm(algorithm=training_algorithm,
                                           input_params=kwargs).fit(X=x, y=y)
        return model

    def _train(self, x, y, classification_algorithm, experiment_start_time,
               split_name):
        model = self._fit(
            x=x,
            y=y,
            training_algorithm=CLASSIFICATION_ALGORITHMS[
                classification_algorithm],
        )

        self._evaluate(
            model=model,
            x=x,
            y=y,
            save_dir=create_save_dir(
                self.config.info_path,
                f"{experiment_start_time}_{classification_algorithm}"),
            report_name=f"train_{split_name}_report.txt")

        save_dir = create_save_dir(
            self.config.model_path,
            f"{experiment_start_time}_{classification_algorithm}")
        self.save_model(model, save_dir)

        save_dir = create_save_dir(
            self.config.info_path,
            f"{experiment_start_time}_{classification_algorithm}")
        self.get_shap(model=model,
                      x=x,
                      classification_algorithm=classification_algorithm,
                      save_dir=save_dir)
        return model

    def _evaluate(self, model, x, y, save_dir, report_name):
        y_pred = model.predict(x)
        report_path = os.path.join(save_dir, report_name)
        text_file = open(report_path, "w")
        text_file.write(
            classification_report(y, y_pred, target_names=list(y.astype(str).unique())))
        text_file.close()

    def start_training(self, get_info: bool = True):

        x = self.dataset.multy_spectral_data
        y = self.dataset.target_data

        rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
        print(f"Split dataset on {rs.get_n_splits(x)} parts")

        print(f"Fitting {self.config.algorithms}")
        experiment_start_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

        for classification_algorithm in tqdm(self.config.algorithms[:]):
            for i, (train_index, test_index) in enumerate(rs.split(X=x)):
                print(f"Training split {i}:")
                model = self._train(
                    x=x.iloc[train_index],
                    y=y.iloc[train_index],
                    classification_algorithm=classification_algorithm,
                    experiment_start_time=experiment_start_time,
                    split_name=f"split_{i}")

                self._evaluate(
                    model=model,
                    x=x.iloc[train_index],
                    y=y.iloc[train_index],
                    save_dir=create_save_dir(
                        self.config.info_path,
                        f"{experiment_start_time}_{classification_algorithm}"),
                    report_name=f"test_split{i}_report.txt")

            print(f"Training on general data:")

            X_train, X_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.33,
                random_state=42,
            )
            model = self._train(
                x=X_train,
                y=y_train,
                classification_algorithm=classification_algorithm,
                experiment_start_time=experiment_start_time,
                split_name="general")

            self._evaluate(
                model=model,
                x=X_test,
                y=y_test,
                save_dir=create_save_dir(
                    self.config.info_path,
                    f"{experiment_start_time}_{classification_algorithm}"),
                report_name=f"test_general_report.txt")

    def get_shap(
        self,
        model,
        x,
        classification_algorithm: str,
        save_dir: str,
    ):
        if classification_algorithm in TREE_ALGORITHMS:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x)

            if shap_values.min() == shap_values.max():
                print(f"bad_data for {save_dir}")
                return

            save_chart(
                method=shap.summary_plot,
                save_path=os.path.join(save_dir, f'shap.png'),
                shap_values=shap_values,
                features=x,
            )

            for i, shap_value in enumerate(shap_values):
                save_chart(
                    method=shap.summary_plot,
                    save_path=os.path.join(save_dir, f'shap_class_{i}.png'),
                    shap_values=shap_value,
                    features=x,
                )

                for j, feature in enumerate(x.columns):
                    save_chart(
                        method=shap.dependence_plot,
                        save_path=os.path.join(
                            save_dir, f'shap_class_{i}_value_{feature}.png'),
                        ind=feature,
                        shap_values=shap_value,
                        features=x,
                        show=False,
                    )


def start_classification(cfg_path: str, get_info=True):
    classification_config = TrainConfig(**get_config_data(cfg_path))
    multy_spectral_data = pd.read_csv(
        classification_config.data_annotation_path)
    multy_spectral_dataset = MultySpectralDataset(
        multy_spectral_data,
        forbidden_features=["image", "class"],
        target="class")
    classification = Classification(multy_spectral_dataset,
                                    classification_config)
    classification.start_training()
