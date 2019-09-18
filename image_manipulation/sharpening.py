import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('imgs\input.jpg')

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')

# Create our sharpening kernel, we don't normalize since the
# the values in the matrix sum to 1
kernel_sharpening = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])

# applying different kernels to the input image
sharpened = cv2.filter2D(img, -1, kernel_sharpening)

axs[1].imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
axs[1].set_title('Image Sharpening')

plt.show()
