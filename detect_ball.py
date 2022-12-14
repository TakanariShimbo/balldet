import argparse
import glob
import os

import cv2
import tqdm
import numpy as np
import pandas as pd


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 50            # Change thresholds
params.filterByColor = True         # Filter by Color.
params.blobColor = 255
params.filterByArea = True          # Filter by Area.
params.minArea = 10
params.filterByCircularity = True   # Filter by Circularity
params.minCircularity = 0.7
params.filterByConvexity = True     # Filter by Convexity
params.minConvexity = 0.8
params.filterByInertia = True       # Filter by Inertia
params.minInertiaRatio = 0.08
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


def detect_ball(opt):
    # save directory
    save_folder_path = f"../datas/balldet_images/{opt.filename}/"
    os.makedirs(save_folder_path, exist_ok=True)

    # read directory
    image_folder_path = f"../datas/original_images/{opt.filename}/"
    mask2_folder_path = f"../datas/mask2_images/{opt.filename}/"
    total_frame = len(glob.glob(image_folder_path+'*.bmp'))

    # ball positions
    ball_positions = []
    for i_frame in tqdm.tqdm(range(total_frame)):
        # read images
        image_path = image_folder_path + f"frame{i_frame}.bmp"
        mask2_path = mask2_folder_path + f"frame{i_frame}.bmp"
        img_bgr = cv2.imread(image_path)
        img_mask2_gray = cv2.imread(mask2_path, 0)

        # detect ball's key points
        key_points = detector.detect(img_mask2_gray)

        image_height, image_width = img_mask2_gray.shape
        if len(key_points) > 0:
            maxval = 0
            for i_keypoint in range(len(key_points)):
                # chose most white point as ball position
                x = int(key_points[i_keypoint].pt[0])
                y = int(key_points[i_keypoint].pt[1])
                val = np.sum(
                    img_bgr[
                    max([y - 3, 0]): min([y + 3, image_height - 1]),
                    max([x - 3, 0]): min([x + 3, image_width - 1]),
                    ]
                )
                if val > maxval:
                    col = x
                    row = y
                    maxval = val
                    key_point = [key_points[i_keypoint]]
        else:
            col = np.nan
            row = np.nan
            key_point = []
        pos = [col/(image_width-1), row/(image_height-1)]
        ball_positions.append(pos)

        # save
        img_mask2_bgr = img_bgr.copy()
        img_mask2_bgr[img_mask2_gray == 0] = 0
        img_res = cv2.addWeighted(img_mask2_bgr, 0.7, img_bgr, 0.3, 1.0)
        img_res = cv2.drawKeypoints(img_res, key_point, np.array([]), (0, 255, 0),
                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        save_path = save_folder_path + f"frame{i_frame}.bmp"
        cv2.imwrite(save_path, img_res)

        # # viz
        # viz
        if opt.show_viz:
            cv2.imshow("balldet", img_res)
            cv2.waitKey(1)

    ball_positions_dataframe = pd.DataFrame(ball_positions, columns=['x', 'y'])
    save_path = save_folder_path + "ball_positions.csv"
    ball_positions_dataframe.to_csv(save_path)


if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--show_viz', type=bool)
    opt = parser.parse_args()
    print(opt)

    # run main
    detect_ball(opt)

    # ex)
    # python detect_ball.py --filename "test_1"