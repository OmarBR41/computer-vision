import cv2
import numpy as np
import os


parameters = {
    'working_dir': '',
    'org': (50, 50),
    'font': cv2.FONT_HERSHEY_SIMPLEX,
    'scale': 1.2,
    'color': (234, 12, 123),
    'thickness': 2,
    'linetype': cv2.LINE_AA
}

# chromakey values
green = {
    'h': 16,
    's': 0,
    'v': 64,
    'h1': 123,
    's1': 111,
    'v1': 187
}

skin_tone = {
    'h': 0,
    's': 74,
    'v': 53,
    'h1': 68,
    's1': 181,
    'v1': 157
}

# amount of data to use
data_size = 1000

# ratio of training data to test data
training_to_test = 0.75

# amount images are scaled down before being fed to keras
img_size = 100

# image height and width (from the webcam)
height, width = 480, 640


def bbox(img):
    """
    :param img: image to read
    :return: region of interest around largest contour
    """
    try:
        bg = np.zeros((1000, 1000), np.uint8)
        bg[250:250+480, 250:250+640] = img
        _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.boundingRect(largest_contour)
        # circ = cv2.minEnclosingCircle(largest_contour)
        x, y, w, h = rect
        x, y = x+w/2, y+h/2
        x, y = x+250, y+250
        ddd = 200
        return bg[y-ddd:y+ddd, x-ddd:x+ddd]
    except:
        return img


def contour_center(c):
    """
    Finds the center of a contour
    :param c: takes a single contour as input
    :return: (x, y) position of the contour's center
    """
    M = cv2.moments(c)
    try:
        center = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    except:
        center = 0, 0
    return center


def in_color_range(img, lower_hsv, upper_hsv):
    """
    Takes image and HSV min-max values; returns parts of image in range

    :param img: image source
    :param lower_hsv: min value of HSV
    :param upper_hsv: max value of HSV
    :return: region of image between lower-upper HSV range
    """
    lh, ls, lv = lower_hsv
    uh, us, uv = upper_hsv
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([lh, ls, lv]), np.array([uh, us, uv])
    mask = cv2.inRange(hsv_img, lower, upper)
    # kernel = np.ones((15, 15), np.uint)
    res = cv2.bitwise_and(img, img, mask=mask)

    return res, mask


def flatten(dimData, images):
    images = np.array(images)
    images = images.reshape(len(images), dimData)
    images = images.astype('float32')
    images /= 255
    return images


def main():
    cap = cv2.VideoCapture(0)

    # get train/test data
    images, labels = [], []
    tool_name = ''
    patterns = []
    tool_num = 0

    while True:
        _, img = cap.read()

        cv2.putText(img, 'Enter class name (then press Enter)', parameters['org'], parameters['font'],
                    parameters['scale'], parameters['color'], parameters['thickness'], parameters['linetype'])
        cv2.putText(img, 'Press Esc when finished', (50, 100), parameters['font'], parameters['scale'],
                    parameters['color'], parameters['thickness'], parameters['linetype'])
        cv2.putText(img, tool_name, (50, 300), parameters['font'], 3, (0, 0, 255), 5, parameters['linetype'])

        cv2.line(img, (330, 240), (310, 240), (234, 123, 234), 3)
        cv2.line(img, (320, 250), (320, 230), (234, 123, 234), 3)

        cv2.imshow('img', img)

        k = cv2.waitKey(1)

        if k > 10:
            tool_name += chr(k)
        elif k == 27:
            break

        current = 0

        if k == 13:
            while current < data_size:
                _, img = cap.read()
                img, mask = in_color_range(img,
                                           (green['h'], green['s'], green['v']),
                                           (green['h1'], green['s1'], green['v1']))
                mask = bbox(mask)
                images.append(cv2.resize(mask, (img_size, img_size)))
                labels.append(tool_num)
                current += 1

                cv2.line(img, (330, 240), (310, 240), (234, 123, 234), 3)
                cv2.line(img, (320, 250), (320, 230), (234, 123, 234), 3)

                cv2.putText(img, 'Collecting data...', parameters['org'], parameters['font'], parameters['scale'],
                            parameters['color'], parameters['thickness'], parameters['linetype'])
                cv2.putText(img, 'Data for {}: {}'.format(tool_name, str(current)), (50, 100), parameters['font'],
                            parameters['scale'], parameters['color'], parameters['thickness'], parameters['linetype'])

                # cv2.imshow('img', img)
                cv2.imshow('mask', mask)

                k = cv2.waitKey(1)
                if k == ord('p'):
                    cv2.waitKey(0)

                if current == data_size:
                    patterns.append(tool_name)
                    tool_name = ''
                    tool_num += 1

                    print(tool_num)
                    break
