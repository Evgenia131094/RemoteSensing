from dataclasses import dataclass


@dataclass
class StartConfig:
    mode: str


@dataclass
class PreprocessingConfig:
    path: str
    annotation: str
    columns: dict
    multyspectral_indexes: list
    info_path: str


@dataclass
class TrainConfig:
    data_annotation_path: str
    algorithms: list
    model_path: str
    info_path: str

