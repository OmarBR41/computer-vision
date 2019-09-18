import cv2
from matplotlib import pyplot as plt


face_classifier = cv2.CascadeClassifier('../../Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('../../Haarcascades/haarcascade_eye.xml')


def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img

    for (x, y, w, h) in faces:
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_classifier.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    roi_color = cv2.flip(roi_color, 1)
    return roi_color


cap = cv2.VideoCapture(0)
v = 0
while True:
    if v == 5:
        break
    ret, frame = cap.read()

    plt.imshow(cv2.cvtColor(face_detector(frame), cv2.COLOR_BGR2RGB))
    plt.title('Our Face Extractor')
    plt.show()

    # cv2.imshow('Our Face Extractor', face_detector(frame))
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break
    v += 1
cap.release()
cv2.destroyAllWindows()
