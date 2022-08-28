import argparse
import glob
import os

import cv2
import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def convert_images2movie(opt):
    # save directory
    save_folder_path = f"../datas/balldet_movies/"
    os.makedirs(save_folder_path, exist_ok=True)

    # read directory
    image_folder_path = f"../datas/balldet_images/{opt.filename}/"
    total_frame = len(glob.glob(image_folder_path+'*.bmp'))

    ball_positions_path = image_folder_path + "ball_positions.csv"
    ball_positions_dataframe = pd.read_csv(ball_positions_path)

    # read ball positions
    idx = ball_positions_dataframe.index.values
    x_pos = ball_positions_dataframe['x'].values
    y_pos = 1.0 - ball_positions_dataframe['y'].values

    # create mp4
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    save_path = save_folder_path + f"{opt.filename}.mp4"
    image_path = image_folder_path + f"frame{0}.bmp"
    img_bgr = cv2.imread(image_path)
    image_height, image_width = img_bgr.shape[:2]
    fig_width = int(image_height / 480 * 640)
    video = cv2.VideoWriter(save_path, fourcc, opt.original_fps/opt.slow_rate, (image_width+fig_width, image_height))

    for i_frame in tqdm.tqdm(range(total_frame)):
        # **** read result images ****
        image_path = image_folder_path + f"frame{i_frame}.bmp"
        img_bgr = cv2.imread(image_path)

        # **** make fig images ****
        # make fig
        fig, ax = plt.subplots()
        # draw line
        ax.plot(idx, x_pos, 'r,-')
        ax.plot(idx, y_pos, 'b,-')
        # draw dot
        ax.scatter([idx[i_frame]], [x_pos[i_frame]], c='r')
        ax.scatter([idx[i_frame]], [y_pos[i_frame]], c='b')
        # set lim
        shift = 5
        ax.set_xlim([i_frame-100+shift, i_frame+shift])
        ax.set_ylim([0, 1])
        # convert fig -> img
        fig.canvas.draw()
        plt.close()
        img_fig_rgba = np.array(fig.canvas.renderer.buffer_rgba())
        img_fig_bgr = cv2.cvtColor(img_fig_rgba, cv2.COLOR_RGBA2BGR)

        # marge imgs
        img_write_bgr = hconcat_resize_min([img_bgr, img_fig_bgr])

        # write img to mp4
        video.write(img_write_bgr)

        # viz
        if opt.show_viz:
            cv2.imshow("Frame", img_write_bgr)
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
    convert_images2movie(opt)

    # ex)
    # python convert_images2movie.py --filename "test_1" --original_fps 30 --slow_rate 4.0


