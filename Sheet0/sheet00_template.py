import cv2 as cv
import numpy as np
import random
import sys
from numpy.random import randint


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img.astype(np.uint8))
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':

    # set image path
    img_path = 'bonn.png'

    # 2a: read and display the image
    img = cv.imread(img_path)
    display_image('2 - a - Original Image', img)

    # 2b: display the intensity image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display_image('2 - b - Intensity Image', img_gray)
    # 2c: for loop to perform the operation
    img_cpy = np.copy(img)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            for chnl in range(img.shape[2]):
                img_cpy[row, col, chnl] = max(
                    0, img_cpy[row, col, chnl] - 0.5 * img_gray[row, col])
    display_image('2 - c - Reduced Intensity Image', img_cpy)

    # 2d: one-line statement to perfom the operation above
    img_cpy = np.uint8(np.maximum(0, np.int16(
        img) - 0.5 * np.expand_dims(img_gray, 2)))
    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)

    # 2e: Extract the center patch and place randomly in the image
    center_y, center_x = int(img.shape[0] / 2), int(img.shape[1] / 2)
    img_patch = img[center_y - 8:center_y + 8, center_x - 8:center_x + 8, :]
    display_image('2 - e - Center Patch', img_patch)

    # Random location of the patch for placement

    # Start with 8 and end with (size - 8) to avoid the 16 x 16 patch from going outside image dimensions
    rand_coord = randint(8, [img.shape[0] - 8, img.shape[1] - 8])
    img_cpy = np.copy(img)
    img_cpy[rand_coord[0] - 8:rand_coord[0] +
            8, rand_coord[1] - 8:rand_coord[1] + 8, :] = img_patch
    display_image('2 - e - Center Patch Placed Random %d, %d' %
                  (rand_coord[0], rand_coord[1]), img_cpy)

    # 2f: Draw random rectangles and ellipses
    img_cpy = np.copy(img)
    for _ in range(10):
        cv.ellipse(img_cpy, tuple(randint(0, [img.shape[1], img.shape[0]])), tuple([randint(img.shape[1] / 2), randint(
            img.shape[0] / 2)]), random.uniform(0, 360), 0, 360, tuple([randint(256), randint(256), randint(256)]), 2)

        cv.rectangle(img_cpy, tuple(randint(0, [img.shape[1], img.shape[0]])), tuple(randint(
            0, [img.shape[1], img.shape[0]])), tuple([randint(256), randint(256), randint(256)]), 2)

    display_image('2 - f - Rectangles and Ellipses', img_cpy)

    # destroy all windows
    cv.destroyAllWindows()
