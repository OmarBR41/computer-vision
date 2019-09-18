import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt


PREDICTOR_PATH = "facial_landmarks/shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()

    for i, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(i), pos, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, (0, 0, 255))
        cv2.circle(im, pos, 3, (0, 255, 255))

    return im


img = cv2.imread('imgs/obama.jpg')
landmarks = get_landmarks(img)
img_landmarks = annotate_landmarks(img, landmarks)

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original")

axs[1].imshow(cv2.cvtColor(img_landmarks, cv2.COLOR_BGR2RGB))
axs[1].set_title("Landmarks")

plt.show()
