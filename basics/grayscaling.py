import cv2
from matplotlib import pyplot as plt


# Load input image
img = cv2.imread('imgs\input.jpg', 1)

# Convert to grayscale with cvtColor
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

row, col = 1, 2
fig, axis = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axis[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axis[0].set_title('Original')

axis[1].imshow(cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB))
axis[1].set_title('Grayscale')

plt.show()
