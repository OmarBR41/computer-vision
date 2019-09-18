import cv2
from matplotlib import pyplot as plt


img = cv2.imread('imgs/shapes4.jpg')

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Shapes')
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, 1)

contours, hierarchy = cv2.findContours(thresh.copy(),
                                       cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)

orig_img = img.copy()

for c in contours:
    # Get approximate polygons
    approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)

    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    if len(approx) == 3:
        shape_name = "Triangle"
        cv2.drawContours(orig_img, [c], 0, (0, 255, 0), -1)

        # Find contour center to place text
        cv2.putText(orig_img, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(c)

        # Check to see if 4-side polygon is square or rectangle
        if abs(w-h) <= 3:
            shape_name = "Square"

            cv2.drawContours(orig_img, [c], 0, (0, 125, 255), -1)
            cv2.putText(orig_img, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        else:
            shape_name = "Rectangle"

            cv2.drawContours(orig_img, [c], 0, (0, 0, 255), -1)
            cv2.putText(orig_img, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    elif len(approx) == 10:
        shape_name = "Star"

        cv2.drawContours(orig_img, [c], 0, (255, 255, 0), -1)
        cv2.putText(orig_img, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    elif len(approx) >= 15:
        shape_name = "Circle"

        cv2.drawContours(orig_img, [c], 0, (0, 255, 255), -1)
        cv2.putText(orig_img, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    plt.title('Identifying Shapes')
    plt.show()
