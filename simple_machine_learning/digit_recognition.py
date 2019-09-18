import cv2
import itertools as it
import numpy as np
import sys
from matplotlib import pyplot as plt

SZ = 20
CLASS_N = 10
PY3 = sys.version_info[0] == 3


def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    if PY3:
        output = it.zip_longest(fillvalue=fillvalue, *args)
    else:
        output = it.izip_longest(fillvalue=fillvalue, *args)
    return output


def mosaic(w, imgs):
    imgs = iter(imgs)
    if PY3:
        img0 = next(imgs)
    else:
        img0 = imgs.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)

    return np.vstack(map(np.hstack, rows))


def split_2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)

    if flatten:
        cells = cells.reshape(-1, sy, sx)

    return cells


def load_digits(fn):
    digits_img = cv2.imread(fn, 0)
    digits = split_2d(digits_img, (SZ, SZ))
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)

    return digits, labels


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    return img


def svm_init(C=12.5, gamma=0.5065):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)

    return model


def svm_train(model, samples, responses):
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    return model


def svm_predict(model, samples):
    return model.predict(samples)[1].ravel()


def svm_evaluate(model, digits, samples, labels):
    predictions = svm_predict(model, samples)
    accuracy = (labels == predictions).mean()
    print("Percentage accuracy: %.2f %%" % (accuracy * 100))

    confusion = np.zeros((10, 10), np.int32)

    for i, j in zip(labels, predictions):
        confusion[int(i), int(j)] += 1

    print("Confusion matrix: ")
    print(confusion)

    vis = []

    for img, flag in zip(digits, predictions == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[..., :2] = 0

        vis.append(img)
    return mosaic(25, vis)


def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ*SZ) / 255.0


def get_hog():
    winSize = (20, 20)
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

    return hog


def main():
    # Load data
    print("Loading digits from 'digits.png'...")
    image = cv2.imread('imgs/digits.png')
    digits, labels = load_digits('imgs/digits.png')

    # Shuffle data
    print("Shuffle data...")
    rand = np.random.RandomState(10)
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    # Deskew images
    print("Deskew images...")
    digits_deskewed = list(map(deskew, digits))

    # HoG feature descriptor
    print("Defining HoG parameters...")
    hog = get_hog()

    print("Calculating HoG descriptor for every image...")
    hog_descriptors = []
    for img in digits_deskewed:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)

    print("Splitting data into training (90%) and test set (10%)...")
    train_n = int(0.9 * len(hog_descriptors))
    digits_train, digits_test = np.split(digits_deskewed, [train_n])
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    print("Training SVM model...")
    model = svm_init()
    svm_train(model, hog_descriptors_train, labels_train)

    print("Evaluating model...")
    vis = svm_evaluate(model, digits_test, hog_descriptors_test, labels_test)

    row, col = 1, 2
    fig, axs = plt.subplots(row, col, figsize=(15, 10))
    fig.tight_layout()

    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Input')

    axs[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Vis')

    plt.show()


if __name__ == '__main__':
    main()
