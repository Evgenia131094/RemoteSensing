import pickle
import os

import cv2
from lxml import objectify
from tqdm import tqdm
import numpy as np


def process_images(model):
    data_path = "Dataset/AMUR_DATA"
    list_of_folders = [name for name in os.listdir(data_path) if "." not in name]

    print("Processing...")
    for folder in tqdm(list_of_folders):
        cur_path = os.path.join(data_path, folder)
        folder = folder.replace(" ", "")
        xml = objectify.parse(os.path.join(cur_path, folder + ".xml"))
        root = xml.getroot()
        imeges_names = os.listdir(cur_path)
        imeges_names.sort()
        images = [cv2.cvtColor(cv2.imread(os.path.join(cur_path, name)), cv2.COLOR_BGR2RGB) for name in imeges_names if
                  '.xml' not in name and '.DS_Store' not in name]
        multilayer_image = np.append(images[0], images[1], axis=2)
        multilayer_image = np.append(multilayer_image, images[2], axis=2)
        multilayer_image = np.append(multilayer_image, images[3], axis=2)
        result_image = multilayer_image[:, :, 1:4]

        for cur_object in root.findall('object'):
            colours = []
            area_class = cur_object.getchildren()[0]
            bbox = list(map(int, cur_object.getchildren()[-1].getchildren()))
            cropped_image = multilayer_image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
            h, w, _ = cropped_image.shape
            for i in range(h):
                for j in range(w):
                    index_y, index_x = bbox[1] + i, bbox[0] + j
                    aerosol = multilayer_image[index_y, index_x][0]
                    blue = multilayer_image[index_y, index_x][1]
                    green = multilayer_image[index_y, index_x][2]
                    red = multilayer_image[index_y, index_x][3]
                    r1 = multilayer_image[index_y, index_x][4]
                    r2 = multilayer_image[index_y, index_x][5]
                    r3 = multilayer_image[index_y, index_x][6]
                    r4 = multilayer_image[index_y, index_x][7]
                    r5 = multilayer_image[index_y, index_x][8]
                    r6 = multilayer_image[index_y, index_x][9]
                    r7 = multilayer_image[index_y, index_x][10]
                    r8 = multilayer_image[index_y, index_x][11]

                    cluster = model.predict([[aerosol, green, red, r1, r2]])[0]
                    if cluster == 0:
                        result_image[bbox[1] + i, bbox[0] + j] = [61, 11, 81]
                    elif cluster == 1:
                        result_image[bbox[1] + i, bbox[0] + j] = [63, 82, 135]
                    elif cluster == 2:
                        result_image[bbox[1] + i, bbox[0] + j] = [70, 143, 139]
                    elif cluster == 3:
                        result_image[bbox[1] + i, bbox[0] + j] = [122, 197, 110]
                    elif cluster == 4:
                        result_image[bbox[1] + i, bbox[0] + j] = [250, 230, 85]


        cv2.imwrite(f"result/{folder}.jpg", cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":

    with open("Clusterization/Models/06_10_2022_14:23:43_KMeans/model.pickle", "rb") as f:
        model = pickle.load(f)

    process_images(model)

