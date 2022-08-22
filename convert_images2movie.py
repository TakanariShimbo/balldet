import argparse
import glob
import os

import cv2
import tqdm
import numpy as np


def main():
    # save directory
    save_folder_path = f"../datas/balldet_movies/"
    os.makedirs(save_folder_path, exist_ok=True)

    # read directory
    image_folder_path = f"../datas/balldet_images/{opt.filename}/"
    total_frame = len(glob.glob(image_folder_path+'*.bmp'))

    # create mp4
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    save_path = save_folder_path + f"{opt.filename}_x{np.round(1/opt.slow_rate, 2)}.mp4"
    image_path = image_folder_path + f"frame{0}.bmp"
    img_bgr = cv2.imread(image_path)
    image_height, image_width = img_bgr.shape[:2]
    video = cv2.VideoWriter(save_path, fourcc, opt.original_fps/opt.slow_rate, (image_width, image_height))

    for i_frame in tqdm.tqdm(range(total_frame)):
        # read images
        image_path = image_folder_path + f"frame{i_frame}.bmp"
        img_bgr = cv2.imread(image_path)

        # write img to mp4
        video.write(img_bgr)

        # viz
        if opt.show_viz:
            cv2.imshow("Frame", img_bgr)
            cv2.waitKey(1)

    video.release()


if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--original_fps', type=int)
    parser.add_argument('--slow_rate', type=float)
    parser.add_argument('--show_viz', type=bool)
    opt = parser.parse_args()
    print(opt)

    # run main
    main()

    # ex)
    # python convert_images2movie.py --filename "test_1" --original_fps 30 --slow_rate 2.0 --show_viz False