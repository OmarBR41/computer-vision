import cv2
import numpy as np


cap = cv2.VideoCapture(0)

_, frame = cap.read()

# setup default location of window
r, h, c, w = 240, 100, 400, 160
track_window = (c, r, w, h)

# crop roi for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# create HSV mask
lower_hsv = np.array([125, 0, 0])
upper_hsv = np.array([175, 255, 255])
mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)

# get color histogram of ROI
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

# normalize values to lie between range 0, 255
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# setup termination criteria
# stop calculating centroid shift after ten iterations or if centroid has moved at least 1 pixel
term_crit = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1

while True:
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # calculate histogram back projection
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # draw it on image
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

        cv2.imshow('Meanshift Tracking', img2)

        if cv2.waitKey(1) == 13:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
