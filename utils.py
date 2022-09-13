import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from canny import CannyDetector
import cv2

mpl.use("Agg")


def remove_background(image):
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = rgb2gray(image)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4
    )
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # opening = cv2.morphologyEx(threshInv, cv2.MORPH_OPEN, kernel, iterations=3)

    # contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # mask = np.zeros(image.shape[:2], np.uint8)
    # largestContour = max(contours, key=cv2.contourArea)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # for c in contours:
    # cv2.drawContours(mask, [c], -1, 255, -1)
    # cv2.drawContours(mask, [largestContour], -1, 255, -1)
    # close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    result = cv2.bitwise_and(original, original, mask=thresh)
    result[thresh == 0] = 255
    return result


def rgb2gray(image):
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return gray


def get_orb_feature_list(img):
    orb = cv2.ORB_create(200, 2.0)
    return orb.detectAndCompute(img, None)


def preprocess_image(image):
    image = remove_background(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = rgb2gray(image)
    temp_list = []
    temp_list.append(image)
    detector = CannyDetector(
        temp_list,
        sigma=1.4,
        kernel_size=5,
        lowThresh=0.06,
        highThresh=0.21,
        weak_pixel=150,
    )
    new_list = detector.detect()
    return new_list[0]


def prepare_images(inputDir):

    image_files = []
    for file in glob.glob(inputDir + "*.jpg", recursive=True):
        image_files.append(file)

    imgs = []

    for i, file in enumerate(image_files):
        image = mpimg.imread(file)
        image = remove_background(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = rgb2gray(image)
        imgs.append(image)
        print("Image Preprocessed: " + str(i + 1))

    detector = CannyDetector(
        imgs,
        sigma=1.4,
        kernel_size=5,
        lowThresh=0.06,
        highThresh=0.21,
        weak_pixel=150,
    )
    new_images = detector.detect()

    for i, img in enumerate(new_images):
        # fig = plt.figure()
        # plt.imshow(img, "gray")
        # plt.savefig(str(image_files[i])[:-4] + "_canny" + str(image_files[i])[-4:])
        # plt.close(fig)
        plt.imsave(
            str(image_files[i])[:-4] + "_canny" + str(image_files[i])[-4:],
            img,
            cmap="gray",
        )
        print("Image Saved: " + str(i + 1))
