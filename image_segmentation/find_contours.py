import cv2
from matplotlib import pyplot as plt


img = cv2.imread('imgs/shapes1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

# Find Canny edges
edges = cv2.Canny(gray, 30, 200)
axs[0].imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
axs[0].set_title('Canny Edges')

# Finding Contours
contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found: {}".format(str(len(contours))))

# Draw contours
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

axs[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[1].set_title('Contours')

plt.show()
