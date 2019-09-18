import cv2
from matplotlib import pyplot as plt


img = cv2.imread('imgs/bottlecaps.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

row, col = 1, 2
fig, axis = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

blur = cv2.medianBlur(gray, 5)

axis[0].imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
axis[0].set_title('Original')
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.5, 10)
print(circles)

for i in circles[0, :]:
    # Outer circle
    # cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)

    # Center of circle
    cv2.circle(img, (i[0], i[1]), 2, (0, 255, 0), 5)

axis[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axis[1].set_title("Detected Circles")

plt.show()
