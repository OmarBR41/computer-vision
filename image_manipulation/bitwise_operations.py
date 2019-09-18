import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('imgs\input.jpg')

# row, col = 1, 2
# fig, axs = plt.subplots(row, col, figsize=(10, 5))
# fig.tight_layout()

square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
# axs[0].imshow(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
# axs[0].set_title('Square Image')

ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
# axs[1].imshow(cv2.cvtColor(ellipse, cv2.COLOR_BGR2RGB))
# axs[1].set_title('Ellipse Image')


row, col = 2, 2
fig, axs = plt.subplots(row, col, figsize=(10, 5))
fig.tight_layout()

# Shows only where they intersect
And = cv2.bitwise_and(square, ellipse)
axs[0][0].imshow(cv2.cvtColor(And, cv2.COLOR_BGR2RGB))
axs[0][0].set_title('And Image')

# Shows where either square or ellipse is
bitwiseOr = cv2.bitwise_or(square, ellipse)
axs[0][1].imshow(cv2.cvtColor(bitwiseOr, cv2.COLOR_BGR2RGB))
axs[0][1].set_title('bitwiseOr Image')

# Shows where either exist by itself
bitwiseXor = cv2.bitwise_xor(square, ellipse)
axs[1][0].imshow(cv2.cvtColor(bitwiseXor, cv2.COLOR_BGR2RGB))
axs[1][0].set_title('bitwiseXor Image')

# Shows everything that isn't part of the square
bitwiseNot_sq = cv2.bitwise_not(square)
axs[1][1].imshow(cv2.cvtColor(bitwiseNot_sq, cv2.COLOR_BGR2RGB))
axs[1][1].set_title('bitwiseNot_sq Image')

plt.show()
