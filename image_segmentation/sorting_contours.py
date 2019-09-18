import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_contour_areas(contours):
    all_areas = []

    for ctr in contours:
        area = cv2.contourArea(ctr)
        all_areas.append(area)

    return all_areas


def label_contour_center(image, c):
    # Places a red circle on the centers of contours
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # Draw the countour number on the image
    cv2.circle(image,(cx,cy), 10, (0,0,255), -1)


def x_cord_contour(contours):
    # Returns the X cordinate for the contour centroid
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return int(M['m10']/M['m00'])
    else:
        pass


def display_left_to_right_contour(i):
    plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))
    plt.title('6 - Left to Right Contour')
    plt.show()


img = cv2.imread('imgs/shapes2.jpg')

row, col = 1, 3
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original")

black_img = np.zeros((img.shape[0], img.shape[1], 3))

copy = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 200)

axs[1].imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
axs[1].set_title("Canny")

contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of contours found: {}".format(str(len(contours))))

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
axs[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[2].set_title("All Contours")

print("Contour areas before sorting: {}".format(str(get_contour_areas(contours))))

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

print("Contour areas after sorting: {}".format(str(get_contour_areas(sorted_contours))))


for c in sorted_contours:
    plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))
    plt.title('Contours by area')
    plt.show()
    cv2.drawContours(copy, [c], -1, (255, 0, 0), 3)

plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))
plt.title('Contours by area')
plt.show()

copy = img.copy()

# Compute centroids and draw them on image
for (i, c) in enumerate(contours):
    label_contour_center(img, c)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Contour Centers')

plt.show()

# Sort by left to right using our x_cord_contour function
contours_left_to_right = sorted(contours, key=x_cord_contour, reverse=False)

# Labeling Contours left to right
for (i, c) in enumerate(contours_left_to_right):
    if i == 0:
        display_left_to_right_contour(i)

    cv2.drawContours(copy, [c], -1, (0, 0, 255), 3)
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(copy, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Draw an approximate
    # rectangle around the binary image
    (x, y, w, h) = cv2.boundingRect(c)

    cropped_contour = copy[y:y + h, x:x + w]