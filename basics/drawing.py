import cv2
import numpy as np
from matplotlib import pyplot as plt


img = np.zeros((512, 512, 3), np.uint8)
# img_bw = np.zeros((512, 512), np.uint8)

# cv2.line(img, (0, 0), (511, 511), (255, 127, 0), 100)
# cv2.rectangle(img, (100, 100), (300, 250), (127, 50, 127), 100)
# cv2.circle(img, (350, 350), 100, (15, 75, 50), 50)

# pts = np.array([[10, 50], [400, 50], [90, 200], [50, 500]], np.int32)
# pts = pts.reshape((-1, 1, 2))
# cv2.polylines(img, [pts], True, (0, 0, 255), 30)

cv2.putText(img, 'Hello World', (75, 290), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 170, 0), 3)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
