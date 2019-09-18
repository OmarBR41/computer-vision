import cv2
from matplotlib import pyplot as plt


img = cv2.imread('imgs\input.jpg')

h, w = img.shape[:2]

# rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), 90, 0.5)
# rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h))

rotated_img = cv2.transpose(img)

# flipped = cv2.flip(img, 1)

plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
# plt.imshow(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))
plt.show()
