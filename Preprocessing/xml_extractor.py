import os

import configparser

from lxml import objectify
import pandas as pd

from PIL import Image

columns = {0: 'image', 1: 'class', 2: 'Aerosol', 3: 'Blue', 4: 'Green', 5: 'Red', 6: 'IR1', 7: 'IR2', 8: 'IR3',
           9: 'IR4', 10: 'IR5', 11: 'IR6',
           12: 'IR7', 13: 'IR8'}


def get_cfg():
    print("Reading config file...")
    config = configparser.ConfigParser()
    config.read(os.environ['CFG_PATH'])
    return config


def get_folders_and_dataframe(data_path, annotation):
    print("Getting folders for preprocessing...")
    processed_images = []
    csv_data = []
    try:
        csv_data = [pd.read_csv(os.path.join(data_path, annotation))]
        processed_images = csv_data[0]["image"].unique()
    except FileNotFoundError:
        pass

    return csv_data, [name for name in list(set(os.listdir(data_path)) - set(processed_images)) if "." not in name]


def get_dataframe_from_image_for_cur_object(cur_object, images, folder):
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
            for j in range(len(colours[0]))]).rename(columns=columns)


def generate_dataset(cfg):
    data_path = cfg["Data"]["path"]
    annotation = cfg["Data"]["annotation"]

    multispectral_data, folders = get_folders_and_dataframe(data_path, annotation)

    print("Start preprocessing...")
    for folder in folders:
        print(f"Processing {folder}")
        cur_path = os.path.join(data_path, folder)
        xml = objectify.parse(os.path.join(cur_path, folder + ".xml"))
        root = xml.getroot()

        imeges_names = os.listdir(cur_path)
        imeges_names.sort()
        images = [Image.open(os.path.join(cur_path, name)) for name in imeges_names if
                  '.xml' not in name and '.DS_Store' not in name]

        for cur_object in root.findall('object'):

            multispectral_data += [get_dataframe_from_image_for_cur_object(cur_object, images, folder)]

    multispectral_data = pd.concat(multispectral_data)
    multispectral_data.dropna()

    print(f"Saving {annotation}")
    multispectral_data.to_csv(os.path.join(data_path, annotation), index=False)


if __name__ == "__main__":
    cfg = get_cfg()
    generate_dataset(cfg)
