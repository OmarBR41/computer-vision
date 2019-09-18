import cv2
import numpy as np


# Initialize capture and store first frame
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('imgs/walking.avi')
_, frame = cap.read()

# create float numpy array with frame values
average = np.float32(frame)

while True:
    _, frame = cap.read()

    # 0.01 is the weight of the image, play around to see how it changes
    cv2.accumulateWeighted(frame, average, 0.01)

    # scale, calculate absolute values and convert result to 8bit
    background = cv2.convertScaleAbs(average)

    cv2.imshow('Input', frame)
    cv2.imshow('Disappearing background', background)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
