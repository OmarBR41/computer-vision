import cv2
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread('imgs/opencv_inv.png', 0)

kernel = np.ones((5, 5), np.uint8)

row, col = 2, 2
fig, axs = plt.subplots(row, col, figsize=(10, 5))
fig.tight_layout()

# Erode
erosion = cv2.erode(image, kernel, iterations=1)
axs[0][0].imshow(cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB))
axs[0][0].set_title('Erosion')

# Dilate
dilation = cv2.dilate(image, kernel, iterations=1)
axs[0][1].imshow(cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB))
axs[0][1].set_title('Dilation')

# Opening - Good for removing noise
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
axs[1][0].imshow(cv2.cvtColor(opening, cv2.COLOR_BGR2RGB))
axs[1][0].set_title('Opening')

# Closing - Good for removing noise
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
axs[1][1].imshow(cv2.cvtColor(closing, cv2.COLOR_BGR2RGB))
axs[1][1].set_title('Closing')


plt.show()
