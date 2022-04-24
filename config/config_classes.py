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
class ClusterizationConfig:
    data_annotation_path: str
    algorithms: list
    info_path: str

