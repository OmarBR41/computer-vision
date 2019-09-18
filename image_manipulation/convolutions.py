import cv2
from matplotlib import pyplot as plt


img = cv2.imread('imgs\elephant.jpg')

# row, col = 1, 2
# fig, axs = plt.subplots(row, col, figsize=(15, 10))
# fig.tight_layout()
#
# # 3x3 Kernel
# kernel_3x3 = np.ones((3, 3), np.float32) / 9
#
# blurred = cv2.filter2D(img, -1, kernel_3x3)
# axs[0].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
# axs[0].set_title('3x3 Kernel Blurring')
#
# # 7x7 Kernel
# kernel_7x7 = np.ones((7, 7), np.float32) / 49
#
# blurred2 = cv2.filter2D(img, -1, kernel_7x7)
# axs[1].imshow(cv2.cvtColor(blurred2, cv2.COLOR_BGR2RGB))
# axs[1].set_title('7x7 Kernel Blurring')


row, col = 2, 2
fig, axs = plt.subplots(row, col, figsize=(10, 5))
fig.tight_layout()

# Averaging done by convolving the image with a normalized box filter.
# This takes the pixels under the box and replaces the central element
# Box size needs to odd and positive
blur = cv2.blur(img, (3, 3))
axs[0][0].imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
axs[0][0].set_title('Averaging')

# Instead of box filter, gaussian kernel
Gaussian = cv2.GaussianBlur(img, (7, 7), 0)
axs[0][1].imshow(cv2.cvtColor(Gaussian, cv2.COLOR_BGR2RGB))
axs[0][1].set_title('Gaussian')

# Takes median of all the pixels under kernel area and central
# element is replaced with this median value
median = cv2.medianBlur(img, 5)
axs[1][0].imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
axs[1][0].set_title('Median')

# Bilateral is very effective in noise removal while keeping edges sharp
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
axs[1][1].imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB))
axs[1][1].set_title('Bilateral')


# dst = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
#
# plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
# plt.title('Fast Means Denoising')


plt.show()
