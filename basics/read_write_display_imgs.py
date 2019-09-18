import cv2
from matplotlib import pyplot as plt


input = cv2.imread('imgs\input.jpg', 1)
plt.imshow(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))
plt.title('Hello World')
plt.show()

# prints height, width and channels of img
print(input.shape)

print("Height of image: {}px".format(input.shape[0]))
print("Width of image: {}px".format(input.shape[1]))

# write output img
# cv2.imwrite('imgs/output.jpg', input)
# cv2.imwrite('imgs/output.png', input)
