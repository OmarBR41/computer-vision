import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('imgs\input.jpg')
M = np.ones(img.shape, dtype='uint8') * 175

row, col = 1, 3
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')

# We use this to add this matrix M, to our image
# Notice the increase in brightness
added = cv2.add(img, M)
axs[1].imshow(cv2.cvtColor(added, cv2.COLOR_BGR2RGB))
axs[1].set_title('Added Image')

# Likewise we can also subtract
# Notice the decrease in brightness
subtracted = cv2.subtract(img, M)
axs[2].imshow(cv2.cvtColor(subtracted, cv2.COLOR_BGR2RGB))
axs[2].set_title('Subtracted')

plt.show()
