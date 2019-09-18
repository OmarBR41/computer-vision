import cv2


# Create our body classifier
car_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_car.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('imgs/cars.avi')

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_classifier.detectMultiScale(gray, 1.4, 2)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow("Cars", frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
