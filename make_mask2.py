import os
import cv2
import numpy as np
import pandas as pd

# setting
filename = 'test_1'

# make directory for save
mask2_folder_path = "../datas/mask2_images/{}/".format(filename)
os.makedirs(mask2_folder_path, exist_ok=True)

for i in range(1000):
    mask_path = '../datas/mask_images/{}/mask{}.bmp'.format(filename, i*4)
    image_path = '../datas/original_images/{}/image{}.bmp'.format(filename, i * 4)
    positions_path = '../datas/humans_positions/{}/image{}.txt'.format(filename, i*4)

    img_mask_gray = cv2.imread(mask_path, 0)
    img_bgr = cv2.imread(image_path)
    positions_dataframe = pd.read_csv(positions_path, delimiter=' ', header=None)

    image_height, image_width = img_mask_gray.shape
    for position_series in positions_dataframe.values:
        x = position_series[1] * image_width
        y = position_series[2] * image_height
        w = position_series[3] * image_width * 1.2
        h = position_series[4] * image_height * 1.2

        p1 = np.array([x - w / 2, y - h / 2], dtype=int)
        p2 = np.array([x + w / 2, y + h / 2], dtype=int)
        cv2.rectangle(img_mask_gray, p1, p2, 0, cv2.FILLED)

    img_mask2_bgr = img_bgr.copy()
    img_mask2_bgr[img_mask_gray == 0] = 0
    img_viz = cv2.addWeighted(img_mask2_bgr, 0.7, img_bgr, 0.3, 1.0)
    cv2.imshow("", img_viz)
    cv2.waitKey(1)

    cv2.imwrite(mask2_folder_path + "mask{}.bmp".format(i*4), img_mask_gray)