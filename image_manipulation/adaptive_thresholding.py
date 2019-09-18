import cv2
from matplotlib import pyplot as plt


img = cv2.imread('imgs/book.jpg', 0)

row, col = 2, 2
fig, axs = plt.subplots(row, col, figsize=(10, 5))
fig.tight_layout()

# Values below 127 goes to 0 (black, everything above goes to 255 (white)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
axs[0][0].imshow(cv2.cvtColor(thresh1, cv2.COLOR_BGR2RGB))
axs[0][0].set_title('Threshold Binary')

# It's good practice to blur images as it removes noise
img = cv2.GaussianBlur(img, (3, 3), 0)

# Using adaptiveThreshold
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
axs[0][1].imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
axs[0][1].set_title('Adaptive Mean Thresholding')

_, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
axs[1][0].imshow(cv2.cvtColor(th2, cv2.COLOR_BGR2RGB))
axs[1][0].set_title("Otsu's Thresholding")

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5,5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
axs[1][1].imshow(cv2.cvtColor(th3, cv2.COLOR_BGR2RGB))
axs[1][1].set_title("Guassian Thresholding")

plt.show()
