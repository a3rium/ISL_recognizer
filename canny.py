#!/usr/bin/python

from pygments import highlight
import utils
import numpy as np
import sys
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage.filters import convolve


class CannyDetector:
    def __init__(
        self,
        imgs,
        sigma=1,
        highThresh=0.15,
        lowThresh=0.05,
        weak_pixel=75,
        strong_pixel=255,
        kernel_size=5,
    ):
        self.imgs = imgs
        self.sigma = sigma
        self.highThresh = highThresh
        self.lowThresh = lowThresh
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.kernel_size = kernel_size
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.imgs_result = []
        return

    # build gaussian kernel to convolve for blurred image
    def gaussian_kernel(self, kernelSize, sigma=1):
        kernelSize = int(kernelSize) // 2
        x, y = np.mgrid[-kernelSize : kernelSize + 1, -kernelSize : kernelSize + 1]
        norm = 1 / (2.0 * np.pi * sigma**2)
        Gauss = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * norm
        return Gauss

    # apply sobel operator
    def sobel_operator(self, img):
        kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        gradX = ndimage.filters.convolve(img, kernelX)
        gradY = ndimage.filters.convolve(img, kernelY)

        Grad = np.hypot(gradX, gradY)
        Grad *= 255.0 / Grad.max()
        Theta = np.arctan2(gradY, gradX)
        return (Grad, Theta)

    # perform non-max supression
    def calc_nmax_supression(self, grad_mag, grad_dir):
        # set up non max matrix
        gridX, gridY = grad_mag.shape
        grid = np.zeros((gridX, gridY), dtype=np.uint8)
        # set up angles
        angle = grad_dir * 180 / np.pi
        angle[angle < 0] += 180

        # check pixels in the direction relative to angle
        for i in range(1, gridX - 1):
            for j in range(1, gridY - 1):
                try:
                    pixel_after = 255
                    pixel_before = 255

                    # for dir = 0 deg
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        pixel_after = grad_mag[i, j + 1]
                        pixel_before = grad_mag[i, j - 1]
                    # for dir = 45 deg
                    elif 22.5 <= angle[i, j] < 67.5:
                        pixel_after = grad_mag[i + 1, j - 1]
                        pixel_before = grad_mag[i - 1, j + 1]
                    # for dir = 90 deg
                    elif 67.5 <= angle[i, j] < 112.5:
                        pixel_after = grad_mag[i + 1, j]
                        pixel_before = grad_mag[i - 1, j]
                    # for dir = 135 deg
                    elif 112.5 <= angle[i, j] < 157.5:
                        pixel_after = grad_mag[i - 1, j - 1]
                        pixel_before = grad_mag[i + 1, j + 1]

                    # update non max matrix
                    if (grad_mag[i, j] >= pixel_before) and (
                        grad_mag[i, j] >= pixel_after
                    ):
                        grid[i, j] = grad_mag[i, j]
                    else:
                        grid[i, j] = 0

                except IndexError as e:
                    pass
        return grid

    # perform thresholding
    def calc_threshold(self, img):
        # calculate the low and hight thresholds from the ratios
        highThresh = img.max() * self.highThresh
        lowThresh = img.max() * self.lowThresh

        # set up result matrix
        x, y = img.shape
        result = np.zeros((x, y), dtype=np.uint8)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        # locate weak and strong pixels
        strong_row, strong_col = np.where(img >= highThresh)
        weak_row, weak_col = np.where((img >= lowThresh) & (img <= highThresh))

        # add values to result
        result[strong_row, strong_col] = strong
        result[weak_row, weak_col] = weak

        return result

    # perform hystersis on thresholded image
    def apply_hystersis(self, img):
        # set up hystersis matrix
        img_row, img_col = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        # check componensts from top to bottom
        top_to_bottom = img.copy()

        for row in range(1, img_row):
            for col in range(1, img_col):
                try:
                    if top_to_bottom[row, col] == weak:
                        if (
                            top_to_bottom[row, col + 1] == 255
                            or top_to_bottom[row, col - 1] == 255
                            or top_to_bottom[row - 1, col] == 255
                            or top_to_bottom[row + 1, col] == 255
                            or top_to_bottom[row - 1, col - 1] == 255
                            or top_to_bottom[row + 1, col - 1] == 255
                            or top_to_bottom[row - 1, col + 1] == 255
                            or top_to_bottom[row + 1, col + 1] == 255
                        ):
                            top_to_bottom[row, col] = 255
                        else:
                            top_to_bottom[row, col] = 0
                except IndexError as e:
                    pass
        # check componensts from bottom to top
        bottom_to_top = img.copy()

        for row in range(img_row - 1, 0, -1):
            for col in range(img_col - 1, 0, -1):
                try:
                    if bottom_to_top[row, col] == weak:
                        if (
                            bottom_to_top[row, col + 1] == 255
                            or bottom_to_top[row, col - 1] == 255
                            or bottom_to_top[row - 1, col] == 255
                            or bottom_to_top[row + 1, col] == 255
                            or bottom_to_top[row - 1, col - 1] == 255
                            or bottom_to_top[row + 1, col - 1] == 255
                            or bottom_to_top[row - 1, col + 1] == 255
                            or bottom_to_top[row + 1, col + 1] == 255
                        ):
                            bottom_to_top[row, col] = 255
                        else:
                            bottom_to_top[row, col] = 0
                except IndexError as e:
                    pass

        # check componensts from right to left
        right_to_left = img.copy()

        for row in range(1, img_row):
            for col in range(img_col - 1, 0, -1):
                try:
                    if right_to_left[row, col] == weak:
                        if (
                            right_to_left[row, col + 1] == 255
                            or right_to_left[row, col - 1] == 255
                            or right_to_left[row - 1, col] == 255
                            or right_to_left[row + 1, col] == 255
                            or right_to_left[row - 1, col - 1] == 255
                            or right_to_left[row + 1, col - 1] == 255
                            or right_to_left[row - 1, col + 1] == 255
                            or right_to_left[row + 1, col + 1] == 255
                        ):
                            right_to_left[row, col] = 255
                        else:
                            right_to_left[row, col] = 0
                except IndexError as e:
                    pass

        # check componensts from left to right
        left_to_right = img.copy()

        for row in range(img_row - 1, 0, -1):
            for col in range(1, img_col):
                try:
                    if left_to_right[row, col] == weak:
                        if (
                            left_to_right[row, col + 1] == 255
                            or left_to_right[row, col - 1] == 255
                            or left_to_right[row - 1, col] == 255
                            or left_to_right[row + 1, col] == 255
                            or left_to_right[row - 1, col - 1] == 255
                            or left_to_right[row + 1, col - 1] == 255
                            or left_to_right[row - 1, col + 1] == 255
                            or left_to_right[row + 1, col + 1] == 255
                        ):
                            left_to_right[row, col] = 255
                        else:
                            left_to_right[row, col] = 0
                except IndexError as e:
                    pass

        # if final image pixel has high value set as strong
        final_img = top_to_bottom + bottom_to_top + right_to_left + left_to_right
        final_img[final_img > strong] = strong

        return final_img

    def detect(self):
        for i, img in enumerate(self.imgs):
            self.img_blurred = convolve(
                img, self.gaussian_kernel(self.kernel_size, self.sigma)
            )
            self.gradientMag, self.theta = self.sobel_operator(self.img_blurred)
            self.nonMaxImg = self.calc_nmax_supression(self.gradientMag, self.theta)
            self.thresholdImg = self.calc_threshold(self.nonMaxImg)
            img_result = self.apply_hystersis(self.thresholdImg)
            self.imgs_result.append(img_result)
            print("Image Processed (Canny): " + str(i + 1))

        return self.imgs_result


def main():
    parser = argparse.ArgumentParser(
        description="This script applies Canny Edge Detection on input images and saves the result as output. Read usage below for parameter options for the algorithm."
    )
    parser.add_argument("i", type=str, help="The input filename.")
    parser.add_argument("o", type=str, help="The output file name.")
    parser.add_argument(
        "-E",
        "--sigma",
        type=float,
        help="The value of sigma for Canny Detector Algorithm. Enter a float. Default = 1.4",
    )
    parser.add_argument(
        "-k",
        "--kernel_size",
        type=int,
        help="The value of kernel size for Canny Detector Algorithm. Enter an integer. Default = 5",
    )
    parser.add_argument(
        "-l",
        "--lowThresh",
        type=float,
        help="The value of low treshold for Canny Detector Algorithm. Enter a float. Default = 0.06",
    )
    parser.add_argument(
        "-t",
        "--highThresh",
        type=float,
        help="The value of high threshold for Canny Detector Algorithm. Enter a float. Default = 0.21",
    )
    parser.add_argument(
        "-w",
        "--weak_pixel",
        type=int,
        help="The value of weak pixel for Canny Detector Algorithm. Enter an integer. Default = 150",
    )
    parser.add_argument(
        "-s",
        "--strong_pixel",
        type=int,
        help="The value of strong pixel for Canny Detector Algorithm. Enter an integer. Default = 255",
    )
    args = parser.parse_args()

    sigma = 1.4
    kernel_size = 5
    lowThresh = 0.06
    highThresh = 0.21
    strong_pixel = 255
    weak_pixel = 150

    if args.sigma:
        sigma = args.sigma
    if args.kernel_size:
        kernel_size = args.kernel_size
    if args.lowThresh:
        lowThresh = args.lowThresh
    if args.highThresh:
        highThresh = args.highThresh
    if args.weak_pixel:
        weak_pixel = args.weak_pixel
    if args.strong_pixel:
        strong_pixel = args.strong_pixel

    image_paths = []
    imgs = []
    image_paths.append(args.i)

    for path in image_paths:
        image = mpimg.imread(path)
        image = utils.rgb2gray(image)
        imgs.append(image)

    detector = CannyDetector(
        imgs,
        sigma=sigma,
        kernel_size=kernel_size,
        lowThresh=lowThresh,
        highThresh=highThresh,
        weak_pixel=weak_pixel,
        strong_pixel=strong_pixel,
    )
    new_images = detector.detect()

    for img in new_images:
        plt.imsave(args.o, img, cmap="gray")


if __name__ == "__main__":
    main()
