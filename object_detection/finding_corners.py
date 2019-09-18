import cv2
import numpy as np
from matplotlib import pyplot as plt


def cornerDetectionDefault():
    harris_corners = cv2.cornerHarris(gray, 3, 3, 0.05)

    # We use dilation of the corner points to enlarge them
    kernel = np.ones((7, 7), np.uint8)
    harris_corners = cv2.dilate(harris_corners, kernel, iterations=2)

    # Threshold for an optimal value, it may vary depending on the image
    temp[harris_corners > 0.025 * harris_corners.max()] = [255, 127, 127]


def cornerDetectionImproved():
    # We specify top 50 corners
    corners = cv2.goodFeaturesToTrack(gray, 49, 0.01, 15)

    for c in corners:
        x, y = c[0]
        x = int(x)
        y = int(y)
        cv2.rectangle(temp, (x-10, y-10), (x+10, y+10), (0, 255, 0), 2)


img = cv2.imread('imgs/chess.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

temp = img.copy()

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original")

# cornerDetectionDefault()
cornerDetectionImproved()

axs[1].imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
axs[1].set_title("Harris Corners")

plt.show()
