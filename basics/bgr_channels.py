import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('imgs\input.jpg')

B, G, R = cv2.split(img)

zeros = np.zeros(img.shape[:2], dtype='uint8')

row, col = 1, 3
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.merge([R, zeros, zeros]))
axs[0].set_title('Red')

axs[1].imshow(cv2.merge([zeros, G, zeros]))
axs[1].set_title('Green')

axs[2].imshow(cv2.merge([zeros, zeros, B]))
axs[2].set_title('Blue')

plt.show()
