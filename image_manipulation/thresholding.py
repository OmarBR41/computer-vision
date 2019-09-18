import cv2
from matplotlib import pyplot as plt\


img = cv2.imread('imgs\gradient.jpg')

row, col = 2, 3
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0][0].set_title('Original Image')

# Values below 127 goes to 0 (black), everything above goes to 255 (white)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
axs[0][1].imshow(cv2.cvtColor(thresh1, cv2.COLOR_BGR2RGB))
axs[0][1].set_title('1 Threshold - Binary')

# Values below 127 go to 255 and values above 127 go to 0 (reverse of above)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
axs[0][2].imshow(cv2.cvtColor(thresh2, cv2.COLOR_BGR2RGB))
axs[0][2].set_title('2 Threshold - Binary Inverse')

# Values above 127 are truncated (held) at 127 (the 255 argument is unused)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
axs[1][0].imshow(cv2.cvtColor(thresh3, cv2.COLOR_BGR2RGB))
axs[1][0].set_title('3 Threshold - TRESH TRUNC')

# Values below 127 go to 0, above 127 are unchanged
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
axs[1][1].imshow(cv2.cvtColor(thresh4, cv2.COLOR_BGR2RGB))
axs[1][1].set_title('4 Threshold - TRESH TOREZO')

# Resever of above, below 127 is unchanged, above 127 goes to 0
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
axs[1][2].imshow(cv2.cvtColor(thresh5, cv2.COLOR_BGR2RGB))
axs[1][2].set_title('5 Threshold - TRESH TOREZO')

plt.show()
