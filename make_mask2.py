import argparse
import glob
import os

import cv2
import tqdm
import numpy as np
import pandas as pd


def main():
    # save directory
    save_folder_path = f"../datas/mask2_images/{opt.filename}/"
    os.makedirs(save_folder_path, exist_ok=True)

    # read directory
    image_folder_path = f"../datas/original_images/{opt.filename}/"
    mask_folder_path = f"../datas/mask_images/{opt.filename}/"
    humans_bbox_folder_path = f"../datas/humans_bbox/{opt.filename}/labels/"
    total_frame = len(glob.glob(image_folder_path+'*.bmp'))

    for i_frame in tqdm.tqdm(range(total_frame)):
        # read images
        image_path = image_folder_path + f"frame{i_frame}.bmp"
        mask_path = mask_folder_path + f"frame{i_frame}.bmp"
        img_mask_gray = cv2.imread(mask_path, 0)
        img_bgr = cv2.imread(image_path)

        # read yolo bbox
        humans_bbox_path = humans_bbox_folder_path + f"frame{i_frame}.txt"
        humans_bbox_dataframe = pd.read_csv(humans_bbox_path, delimiter=' ', header=None)

        image_height, image_width = img_mask_gray.shape
        for position_series in humans_bbox_dataframe.values:
            # convert bbox yolo -> cv2
            x = position_series[1] * image_width
            y = position_series[2] * image_height
            w = position_series[3] * image_width * 1.2
            h = position_series[4] * image_height * 1.2
            p1 = np.array([x - w / 2, y - h / 2], dtype=int)
            p2 = np.array([x + w / 2, y + h / 2], dtype=int)

            # mask bbox
            cv2.rectangle(img_mask_gray, p1, p2, 0, cv2.FILLED)

        # save
        cv2.imwrite(save_folder_path + f"frame{i_frame}.bmp", img_mask_gray)

        # viz
        if opt.show_viz:
            img_mask2_bgr = img_bgr.copy()
            img_mask2_bgr[img_mask_gray == 0] = 0
            img_viz = cv2.addWeighted(img_mask2_bgr, 0.7, img_bgr, 0.3, 1.0)
            cv2.imshow("Masked Frame v2", img_viz)
            cv2.waitKey(1)


if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--show_viz', type=bool)
    opt = parser.parse_args()
    print(opt)

    # run main
    main()

    # ex)
    # python make_mask2.py --filename "test_1" --show_viz False