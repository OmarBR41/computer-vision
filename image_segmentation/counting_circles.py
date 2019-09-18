import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('imgs/blobs.jpg', 0)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.show()

# Initialize the detector using default parameters
detector = cv2.SimpleBlobDetector_create()

# Detect blobs
keypoints = detector.detect(img)

# Draw blobs on img as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(img, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Total number of blobs: {}".format(str(len(keypoints)))

cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original")

axs[1].imshow(cv2.cvtColor(blobs, cv2.COLOR_BGR2RGB))
axs[1].set_title("Blobs (default parameters)")

plt.show()

# Set filtering parameters
params = cv2.SimpleBlobDetector_Params()

# Set area
params.filterByArea = True
params.minArea = 100

# Set circularity
params.filterByCircularity = True
params.minCircularity = 0.9

# Set convexity
params.filterByConvexity = False
params.minConvexity = 0.2

# Set inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector2 = cv2.SimpleBlobDetector_create(params)

# Detect new blobs
keypoints2 = detector2.detect(img)

# Draw blobs on our image as red circles
blank2 = np.zeros((1, 1))
blobs2 = cv2.drawKeypoints(img, keypoints2, blank2, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs2 = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints2))
cv2.putText(blobs2, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

# Show blobs
row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')

axs[1].imshow(cv2.cvtColor(blobs2, cv2.COLOR_BGR2RGB))
axs[1].set_title('Filtering Circular Blobs Only')

plt.show()
