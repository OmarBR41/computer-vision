import cv2
import numpy as np
from matplotlib import pyplot as plt


# map first digit of credit card number to credit card type
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


def x_cord_contour(contours):
    # Returns the X coordinate for the contour centroid
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return int(M['m10']/M['m00'])
    else:
        pass


def display_ltr_contours(img, contours):
    """
    Takes an array of previously sorted contours and draws them in the image input

    :param img: image to write on and display
    :param contours: array of contours to show (sorted from left to right)
    """
    for (i, c) in enumerate(contours):
        cv2.drawContours(img, [c], -1, (0, 0, 255), 3)
        x, y, w, h = cv2.boundingRect(c)
        cv2.putText(img, str(i), (int(x+w/2), int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def main():
    # load reference image to compare for 0-9 digits
    ref = cv2.imread('imgs/ocr_a_reference.png')

    # pre-process reference image to read its contours and store each digit's RoI
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    _, ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=x_cord_contour)

    # display_ltr_contours(ref, contours)

    digits = {}

    for (i, c) in enumerate(contours):
        # compute bounding box for every digit and store its RoI in 'digits' dict
        x, y, w, h = cv2.boundingRect(c)
        roi = ref[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57, 88))

        digits[i] = roi

    # initialize rectangular and square kernels
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # load input image
    image = cv2.imread('imgs/credit_card_01.png')
    # image = cv2.imread('imgs/credit_card_02.png')
    # image = cv2.imread('imgs/credit_card_03.png')
    h, w = image.shape[:2]
    image = cv2.resize(image, (300, int(h * 300 / w)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply tophat morph
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

    # compute Scharr gradient
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    minVal, maxVal = np.min(gradX), np.max(gradX)
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype('uint8')

    # apply closing operation with rect kernel to close gaps between digits
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    _, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    # find contours in thresholded image
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # digit locations
    locs = []

    for (i, c) in enumerate(cnts):
        # compute bounding box of every contour and get aspect ratio
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)

        # credit cards use a fixed size fonts with 4 groups of 4 digits
        if 2.5 < ar < 4.0:
            if 40 < w < 55 and 10 < h < 20:
                locs.append((x, y, w, h))

    # sort locations from left to right
    locs = sorted(locs, key=lambda x: x[0])
    output = []

    # loop over the 4 groups of 4 digits
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # initialize the list of group digits
        groupOutput = []

        # extract RoI of 4 digits from preprocessed image
        group = gray[gY-5:gY+gH+5, gX-5:gX+gW+5]
        _, group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # get contors of each digit in group, then sort left to right
        digitContours, _ = cv2.findContours(group, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digitContours = sorted(digitContours, key=x_cord_contour)

        # loop over digits
        for c in digitContours:
            # compute bounding box of digit, extract ROI and resize it to match reference size
            x, y, w, h = cv2.boundingRect(c)
            roi = group[y:y+h, x:x+w]
            roi = cv2.resize(roi, (57, 88))

            # list of template matching scores
            scores = []

            # loop over the reference digit name and digit ROI
            for (digit, digitROI) in digits.items():
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)

                _, score, _, _ = cv2.minMaxLoc(result)

                scores.append(score)

            # the classification for the digit ROI will be the reference digit name
            # with the highest template matching score
            groupOutput.append(str(np.argmax(scores)))

        # draw the digit classifications around the group
        cv2.rectangle(image, (gX-5, gY-5), (gX+gW+5, gY+gH+5), (0, 0, 255), 2)
        cv2.putText(image, "".join(groupOutput), (gX, gY-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        output.extend(groupOutput)

    print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
    print("Credit Card #: {}".format("".join(output)))

    cv2.imshow('img', image)
    cv2.waitKey(0)


if '__main__' == __name__:
    main()
