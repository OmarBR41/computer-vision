import cv2
from matplotlib import pyplot as plt


img = cv2.imread('imgs/house.jpg')
orig_img = img.copy()

plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
plt.title('House')
plt.show()

# Grayscale and binarization
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Iterate through each contour and compute the bounding rectangle
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(orig_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    plt.title('Bounding Rectangle')
    plt.show()

# Iterate through each contour and compute the approx contour
for c in contours:
    # Calculate accuracy as percent of the contour perimeter
    accuracy = 0.03 * cv2.arcLength(c, True)
    approximation = cv2.approxPolyDP(c, accuracy, True)
    cv2.drawContours(img, [approximation], 0, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Approx Poly DP')
    plt.show()
