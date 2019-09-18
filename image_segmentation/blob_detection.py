import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image
image = cv2.imread("imgs/sunflowers.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()

# Detect blobs.
keypoints = detector.detect(image)

# Draw detected blobs as red circles.
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(gray, keypoints, blank, (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')

axs[1].imshow(cv2.cvtColor(blobs, cv2.COLOR_BGR2RGB))
axs[1].set_title('Output_Blobs')

plt.show()
