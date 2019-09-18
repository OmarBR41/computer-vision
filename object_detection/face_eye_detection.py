import cv2
import numpy as np
from matplotlib import pyplot as plt


def face_detection():
    # Loads our XML classifier
    face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    # Return the ROI of the detected face as tuple, storing the top-left and bottom-right coordinates
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # When no faces detected, returns an empty tuple
    if faces is ():
        print("No faces found")

    # else, we iterate through our faces array and draw its respective bounding rect
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (127, 0, 255), 2)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Face Detection')
        plt.show()

        roi = img[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 2)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Eye Detection')
            plt.show()


img = cv2.imread('imgs/trump.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.show()

face_detection()
