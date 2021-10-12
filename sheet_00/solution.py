import cv2 as cv
import numpy as np
import random
import sys
from numpy.random import randint


def display_image(window_name: str, img: np.ndarray):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def reduced_intensity(img: np.ndarray):
    """
    Substract the reduced intensity image of `img` from each channel by the formula:
    `max(channel-intensity*0.5, 0)`. This is an in-place operation
    :param img: image object to reduce the intensity
    """
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ri = img_gray[i][j] * 0.5
            for k in range(img.shape[2]):
                img[i][j][k] = max(img[i][j][k]-ri, 0)


COLORS = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255),
          (255, 0, 0), (255, 0, 255), (255, 255, 0), (255, 255, 255)]
MIN_SHAPE_SIZE, MAX_SHAPE_SIZE = 20, 50


def draw_rectangle(img: np.ndarray):
    """
    Draw a rectangle on the `img` with random dimensions in the interval of
    `(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)` and fill it with random color sampled from `COLORS`.
    Pick a random position for the rectangle.
    :param img: image object to draw a rectangle on
    """
    img_height, img_width = img.shape[:2]
    rect_height = randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
    rect_width = randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
    x0, y0 = randint(img_width-rect_width), randint(img_height-rect_height)
    color = random.choice(COLORS)
    cv.rectangle(img, (x0, y0), (x0+rect_width, y0+rect_height), color, -1)


def draw_ellipse(img: np.ndarray):
    """
    Draw an ellipse on the `img` with random dimensions of the axes in the interval of
    `(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)` and fill it with random color sampled from `COLORS`.
    Pick a random position for the ellipse and a random angle.
    :param img: image object to draw an ellipse on
    """
    img_height, img_width = img.shape[:2]
    axis1, axis2 = randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE), randint(
        MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
    major_axis = max(axis1, axis2)
    x, y = randint(major_axis, img_width - major_axis), randint(major_axis, img_height-major_axis)
    color = random.choice(COLORS)
    angle = randint(360)
    cv.ellipse(img, (x, y), (axis1, axis2), angle, 0, 360, color, -1)


if __name__ == '__main__':

    # set image path
    IMG_PATH = '../bonn.png'

    # 2a: read and display the image
    img = cv.imread(IMG_PATH)
    display_image('2 - a - Original Image', img)

    # 2b: display the intensity image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display_image('2 - b - Intensity Image', img_gray)

    # 2c: for loop to perform the operation
    img_cpy = img.copy()
    reduced_intensity(img_cpy)
    display_image('2 - c - Reduced Intensity Image', img_cpy)

    # 2d: one-line statement to perfom the operation above
    img_cpy = np.maximum(img-np.expand_dims(img_gray, 2) * 0.5, 0).astype(np.uint8)
    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)

    # 2e: Extract the center patch and place randomly in the image
    y, x = int(img.shape[0]/2), int(img.shape[1]/2)
    img_patch = img[y-8:y+8, x-8:x+8]
    display_image('2 - e - Center Patch', img_patch)

    # Random location of the patch for placement
    img_cpy = img.copy()
    img_height, img_width = img.shape[:2]
    y0, x0 = randint(img_height-16), randint(img_width-16)
    img_cpy[y0:y0+16, x0:x0+16] = img_patch
    rand_coord = (x0, y0)
    display_image(f'2 - e - Center Patch Placed Random {rand_coord}', img_cpy)

    # 2f: Draw random rectangles and ellipses
    img_cpy = img.copy()
    for _ in range(10):
        draw_rectangle(img_cpy)
        draw_ellipse(img_cpy)
    display_image('2 - f - Rectangles and Ellipses', img_cpy)

    # destroy all windows
    cv.destroyAllWindows()
