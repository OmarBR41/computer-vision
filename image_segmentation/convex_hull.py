import cv2
from matplotlib import pyplot as plt


img = cv2.imread('imgs/hand.jpg')
orig_img = img.copy()

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Hand')
plt.show()

# Preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresh
ret, thresh = cv2.threshold(gray, 176, 255, 0)

# Contours
contours, hierarchy = cv2.findContours(thresh.copy(),
                                       cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)

# Draw contours
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(orig_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    plt.title('Bounding Rectangle')
    plt.show()

# Sort contours by area and remove the largest frame contour
n = len(contours) - 1
contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]

# Iterate through contours and draw the convex hull
for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(img, [hull], 0, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Convex Hull')
    plt.show()
