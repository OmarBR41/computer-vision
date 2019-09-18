import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('imgs/ex2.jpg')
h, w, c = img.shape

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original')

# Coordinates of the 4 corners of the original img
ptsA = np.float32([[320, 15], [700, 215], [85, 610]])

# Coordinates of the 4 corners of the desired output
ptsB = np.float32([[0, 0], [420, 0], [0, 594]])

M = cv2.getAffineTransform(ptsA, ptsB)

warped = cv2.warpAffine(img, M, (w, h))

axs[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
axs[1].set_title('Warp Perspective')

plt.show()
