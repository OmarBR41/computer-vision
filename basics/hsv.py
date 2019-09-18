import cv2
from matplotlib import pyplot as plt


img = cv2.imread('imgs\input.jpg')

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

row, col = 2, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0][0].imshow(cv2.cvtColor(hsv_img, cv2.COLOR_BGR2RGB))
axs[0][0].set_title('HSV image')

axs[0][1].imshow(cv2.cvtColor(hsv_img[:, :, 0], cv2.COLOR_BGR2RGB))
axs[0][1].set_title('Hue channel')

axs[1][0].imshow(cv2.cvtColor(hsv_img[:, :, 1], cv2.COLOR_BGR2RGB))
axs[1][0].set_title('Saturation channel')

axs[1][1].imshow(cv2.cvtColor(hsv_img[:, :, 2], cv2.COLOR_BGR2RGB))
axs[1][1].set_title('Value channel')

plt.show()
