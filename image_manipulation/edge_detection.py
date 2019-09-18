import cv2
from matplotlib import pyplot as plt

image = cv2.imread('imgs/input.jpg')

# Extract Sobel Edges
sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
laplacian = cv2.Laplacian(image, cv2.CV_64F)

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')

# Then, we need to provide two values: threshold1 and threshold2. Any gradient
# value larger than threshold2 is considered to be an edge. Any value below
# threshold1 is considered not to be an edge. Values in between threshold1 and
# threshold2 are either classiﬁed as edges or non-edges based on how their
# intensities are “connected”. In this case, any gradient values below 60 are
# considered non-edges whereas any values above 120 are considered edges.

# Canny Edge Detection uses gradient values as thresholds
# The first threshold gradient
canny = cv2.Canny(image, 50, 120)
axs[1].imshow(cv2.cvtColor(canny, cv2.COLOR_BGR2RGB))
axs[1].set_title('canny')

plt.show()
