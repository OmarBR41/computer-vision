import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('imgs/waldo1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title("Where is Waldo?")

axs[1].imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
axs[1].set_title("Gray Waldo beach")

plt.show()

gray_waldo = cv2.imread('imgs/waldo2.jpg', 0)
color_waldo = cv2.imread('imgs/waldo2.jpg')

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(gray_waldo, cv2.COLOR_BGR2RGB))
axs[0].set_title("Gray Waldo")

axs[1].imshow(cv2.cvtColor(color_waldo, cv2.COLOR_BGR2GRAY))
axs[1].set_title("Color Waldo")

result = cv2.matchTemplate(gray, gray_waldo, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Create bounding rect
top_left = max_loc
bottom_right = (top_left[0] + 50, top_left[1] + 50)
cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 5)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Where is Waldo?')
plt.show()
