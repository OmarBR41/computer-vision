import cv2
from matplotlib import pyplot as plt


img = cv2.imread('imgs\input.jpg')
h, w = img.shape[:2]

start_row, start_col = int(h * .25), int(w * .25)
end_row, end_col = int(h * .75), int(w * .75)

cropped = img[start_row:end_row, start_col:end_col]

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')

axs[1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
axs[1].set_title('Cropped Image')

plt.show()
