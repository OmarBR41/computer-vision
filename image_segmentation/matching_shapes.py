import cv2
from matplotlib import pyplot as plt


# Load the shape template or reference image
template = cv2.imread('imgs/star.jpg', 0)

plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
plt.title('Star')
plt.show()

# Load the target image with the shapes we're trying to match
target = cv2.imread('imgs/shapes3.jpg')
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

plt.imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
plt.title('Target')
plt.show()

# Threshold first image
ret, thresh1 = cv2.threshold(template, 127, 255, 0)

# Find contours in template
contours, hierarchy = cv2.findContours(thresh1,
                                       cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_SIMPLE)

# Sort contours to remove the one with largest area and extract the second largest
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
template_contour = sorted_contours[1]

# Threshold second image
ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh2,
                                       cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    match = cv2.matchShapes(template_contour, c, 3, 0.0)
    print(match)

    closest_contour = c if match < 0.15 else []

cv2.drawContours(target, [closest_contour], -1, (0, 255, 0), 3)

plt.imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
plt.title('Output')
plt.show()
