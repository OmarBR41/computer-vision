import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('imgs\input.jpg')

h, w = img.shape[:2]
quarter_h, quarter_w = h/4, w/4

T = np.float32([[1, 0, quarter_w],
                [0, 1, quarter_h]])

img_translation = cv2.warpAffine(img, T, (w, h))

plt.imshow(cv2.cvtColor(img_translation, cv2.COLOR_BGR2RGB))
plt.show()
