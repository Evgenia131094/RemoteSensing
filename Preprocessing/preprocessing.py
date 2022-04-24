from config.config_classes import PreprocessingConfig
from Utils.utils import get_config_data
from Dataset.dataset import MultySpectralDataset


import os

from lxml import objectify
import pandas as pd

from PIL import Image

from tqdm import tqdm


def get_converted_multyspectral_data_and_list_of_remained_folders(data_path: str, annotation: str):
    print("Getting folders for preprocessing...")
    processed_images = []
    csv_data = []
    try:
        csv_data = [pd.read_csv(os.path.join(data_path, annotation))]
        processed_images = csv_data[0]["image"].unique()
    except FileNotFoundError:
        pass

    return csv_data, [name for name in list(set(os.listdir(data_path)) - set(processed_images)) if "." not in name]


def get_dataframe_from_image_for_cur_object(cur_object, images, folder) -> pd.DataFrame:
    colours = []
    area_class = cur_object.getchildren()[0]
    bbox = list(map(int, cur_object.getchildren()[-1].getchildren()))
    for i, image in enumerate(images):
        colours.append(list(image.crop((bbox[0], bbox[1], bbox[2], bbox[3])).convert("RGB").getdata()))
    return pd.DataFrame([[folder, area_class] +
                         list(colours[0][j] +
                              colours[1][j] +
                              colours[2][j] +
                              colours[3][j])
                         for j in range(len(colours[0]))])


def generate_dataset(preprocessing_config) -> pd.DataFrame:
    multyspectral_data, list_of_remained_folders = get_converted_multyspectral_data_and_list_of_remained_folders(
        preprocessing_config.path,
        preprocessing_config.annotation)

    print("Converting xml to pandas dataframe...")
    for folder in tqdm(list_of_remained_folders):
        cur_path = os.path.join(preprocessing_config.path, folder)
        xml = objectify.parse(os.path.join(cur_path, folder + ".xml"))
        root = xml.getroot()

        imeges_names = os.listdir(cur_path)
        imeges_names.sort()
        images = [Image.open(os.path.join(cur_path, name)) for name in imeges_names if
                  '.xml' not in name and '.DS_Store' not in name]

        for cur_object in root.findall('object'):
            multyspectral_data += [get_dataframe_from_image_for_cur_object(cur_object, images, folder).rename(
                                                                           columns=preprocessing_config.columns)]

    print("All data  preprocessed!")
    multyspectral_data = pd.concat(multyspectral_data)
    multyspectral_data.dropna()

    print(f"Saving {preprocessing_config.annotation} with {len(multyspectral_data)} samples")
    saving_path = os.path.join(preprocessing_config.path, preprocessing_config.annotation)
    multyspectral_data.to_csv(saving_path, index=False)
    return multyspectral_data


def get_data_from_xml(preprocessing_config) -> pd.DataFrame:
    print("Getting dataframe from xml...")
    return generate_dataset(preprocessing_config)


def get_dataset_info(multy_spectral_data: pd.DataFrame, info_path: str):
    multy_spectral_dataset = MultySpectralDataset(multy_spectral_data, info_path)
    multy_spectral_dataset.get_dataset_info_in_charts()


def start_preprocessing(cfg_path: str, get_info: bool):
    preprocessing_config = PreprocessingConfig(**get_config_data(cfg_path))
    print("Start preprocessing...")
    multy_spectral_data = get_data_from_xml(preprocessing_config)

    if get_info:
        print("Getting dataset info...")
        get_dataset_info(multy_spectral_data, preprocessing_config.info_path)

