import cv2
import numpy as np


# initialize webcam
cap = cv2.VideoCapture(0)

# define range of HSV values (purple, in this case)
lower_hsv = np.array([45, 0, 0])
upper_hsv = np.array([75, 255, 255])

while True:
    _, frame = cap.read()

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Filtered Color', res)

    if cv2.waitKey(1) == 13: # if Enter key is pressed
        break

cap.release()
cv2.destroyAllWindows()
