import cv2


def canny(img):
    # Convert img to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur img
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Edges
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)

    # Invert binarize img
    _, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)

    return mask


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.imshow('Canny Stream', canny(frame))

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
