import cv2
from matplotlib import pyplot as plt


img = cv2.imread('imgs\input.jpg')

histogram = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.hist(img.ravel(), 256, [0, 256])
plt.show()

color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histogram2 = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histogram2, color=col)
    plt.xlim([0, 256])

plt.show()
