import cv2
from matplotlib import pyplot as plt


img = cv2.imread('imgs\input.jpg')

smaller = cv2.pyrDown(img)
larger = cv2.pyrUp(smaller)

row, col = 1, 3
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')

axs[1].imshow(cv2.cvtColor(smaller, cv2.COLOR_BGR2RGB))
axs[1].set_title('Smaller Image')

axs[2].imshow(cv2.cvtColor(larger, cv2.COLOR_BGR2RGB))
axs[2].set_title('Larger Image')

plt.show()
