import cv2
import numpy as np

# setting
filename = "test_1"

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 50            # Change thresholds
params.filterByColor = True         # Filter by Color.
params.blobColor = 255
params.filterByArea = True          # Filter by Area.
params.minArea = 30
params.filterByCircularity = True   # Filter by Circularity
params.minCircularity = 0.7
params.filterByConvexity = True     # Filter by Convexity
params.minConvexity = 0.8
params.filterByInertia = True       # Filter by Inertia
params.minInertiaRatio = 0.08
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


# main
for i in range(1000):
    image_path = '../datas/original_images/{}/image{}.bmp'.format(filename, i * 4)
    mask2_path = '../datas/mask2_images/{}/mask{}.bmp'.format(filename, i*4)

    img_bgr = cv2.imread(image_path)
    img_mask2_gray = cv2.imread(mask2_path, 0)
    img_mask2_bgr = img_bgr.copy()
    img_mask2_bgr[img_mask2_gray == 0] = 0
    image_height, image_width = img_mask2_gray.shape
    key_points = detector.detect(img_mask2_gray)

    # 一番白っぽいところをボールの位置とする
    if len(key_points) > 0:
        maxval = 0
        for i in range(len(key_points)):
            x = int(key_points[i].pt[0])
            y = int(key_points[i].pt[1])
            val = np.sum(
                img_mask2_bgr[
                max([y - 3, 0]): min([y + 3, image_height - 1]),
                max([x - 3, 0]): min([x + 3, image_width - 1]),
                ]
            )
            if val > maxval:
                col = x
                row = y
                maxval = val
                key_point = [key_points[i]]
    else:
        col = 0
        row = 0
        key_point = []
    pos = np.array([col, row])

    img_viz = cv2.addWeighted(img_mask2_bgr, 0.7, img_bgr, 0.3, 1.0)
    img_viz = cv2.drawKeypoints(img_viz, key_point, np.array([]), (0, 255, 0),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("balldet", img_viz)
    cv2.waitKey(1)

