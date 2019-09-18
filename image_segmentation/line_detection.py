import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('imgs/sudoku.jpg')

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Sudoku')
plt.show()

orig_img = img.copy()

# Grayscale and Canny pre-processing
gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 170, apertureSize=3)


def houghLines():
    # Run HoughLines using a rho accuracy of 1px, theta accuracy of np.pi / 180 (1 degree) and line threshold of 240
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 240)

    # Iterate and convert each line
    for line in lines:
        rho, theta = zip(line[0])
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv2.line(orig_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    plt.title('Hough Lines')
    plt.show()


def probHoughLines():
    probLines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, 5, 10)
    print(probLines)

    for line in probLines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]

        cv2.line(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    plt.title('Probabilistic Hough Lines')
    plt.show()


probHoughLines()
