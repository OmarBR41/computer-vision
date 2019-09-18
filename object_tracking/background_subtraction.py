import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('imgs/walking.avi')

# bg subtraction MOG2 setup
foreground_background = cv2.createBackgroundSubtractorMOG2()

# bg subtraction KNN setup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorKNN()


def bgMOG2(frame):
    foreground_mask = foreground_background.apply(frame)
    cv2.imshow('Output', foreground_mask)


def bgKNN(frame):
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('Output', fgmask)


while True:
    _, frame = cap.read()

    # bgMOG2(frame)
    bgKNN(frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
