import cv2 as cv
import numpy as np
import random
import time


def print_task_name(task_name: str):
    """Helper function for making better terminal logging"""
    print(f"\n{task_name}\n{'-'*len(task_name)}")


def display_imgs(im_dict: dict):
    """Helper function for displaying images"""
    for window_name, img in im_dict.items():
        cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# ********************TASK1***********************


def integral_image(img: np.ndarray) -> np.ndarray:
    """Return integral image from the given image.
    The method assumes grayscale image as input"""
    height, width = img.shape
    integral = np.zeros(shape=(height+1, width+1), dtype=np.int32)
    for i in range(1, height+1):
        for j in range(1, width+1):
            integral[i][j] = img[i-1][j-1] + integral[i-1][j] + \
                integral[i][j-1] - integral[i-1][j-1]
    return integral


def sum_image(image: np.ndarray) -> int:
    """Return the sum of pixel values of the input image. 
    The method assumes grayscale image as input"""
    pixel_sum = 0
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            pixel_sum += image[i][j]
    return pixel_sum


def task1():
    print_task_name("1. Rectangles and Integral Images")
    img = cv.imread('bonn.png', cv.IMREAD_GRAYSCALE)
    height, width = img.shape
    # 1.a
    integral_custom = integral_image(img)
    integral_custom = integral_custom / integral_custom.max() * 255
    integral_custom = integral_custom.astype(np.uint8)
    display_imgs({'integral custom': integral_custom})

    # 1.b and 1.c
    rand_pts = [(np.random.randint(width-100), np.random.randint(height-100))
                for _ in range(10)]
    rand_rects = [((x, y), (x+100, y+100)) for (x, y) in rand_pts]
    means_cv = []
    means_custom = []
    means_summing = []

    # cv.integral
    start = time.perf_counter()
    integral_cv = cv.integral(img)
    for ((x0, y0), (x1, y1)) in rand_rects:
        int_rect_sum = (integral_cv[y1][x1] - integral_cv[y0][x1] -
                        integral_cv[y1][x0] + integral_cv[y0][x0])
        int_rect_mean = int_rect_sum / 10000.0
        means_cv.append(int_rect_mean)
    end = time.perf_counter()
    print(f'cv.integral time: {end-start:.5f}')

    # custom integral
    start = time.perf_counter()
    integral_custom = integral_image(img)
    for ((x0, y0), (x1, y1)) in rand_rects:
        int_rect_sum = (integral_custom[y1][x1] - integral_custom[y0][x1] -
                        integral_custom[y1][x0] + integral_custom[y0][x0])
        int_rect_mean = int_rect_sum / 10000.0
        means_custom.append(int_rect_mean)
    end = time.perf_counter()
    print(f'custom integral time: {end-start:.5f}')

    # mean value by summing
    start = time.perf_counter()
    for ((x0, y0), (x1, y1)) in rand_rects:
        rect = img[y0:y1, x0:x1]
        rect_sum = sum_image(rect)
        rect_mean = rect_sum / 10000.0
        means_summing.append(rect_mean)
    end = time.perf_counter()
    print(f'mean value by summing: {end-start:.5f}')
    print(f'Are the three lists the same: {means_custom == means_cv == means_summing}')


task1()

# ************************************************
# ********************TASK2***********************


def equalize_hist_image(img: np.ndarray):
    # Your implementation of histogram equalization
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    hist_norm = hist / hist.sum()
    cdf = hist_norm.cumsum()
    equalized_hist = cdf * 255
    equalized_img = equalized_hist[img].astype(np.uint8)
    return equalized_img


def task2():
    print_task_name('2. Histogram Equalization')
    img = cv.imread('bonn.png', cv.IMREAD_GRAYSCALE)
    equalized_cv = cv.equalizeHist(img)
    equalized_custom = equalize_hist_image(img)
    display_imgs({
        'original': img,
        'cv.equalizedHist': equalized_cv,
        'custom implementation of cv.equalizedHist': equalized_custom
    })
    pix_err = np.abs(equalized_custom-equalized_cv)
    print(f'Maximum pixel error: {pix_err.max()}')


# task2()


# ************************************************
# ********************TASK4***********************


def get_kernel(sigma: float) -> np.ndarray:
    """Return 1D Gaussian kernel of shape (ksize, 1) where ksize is calculated
    according to the rule of thumb from the lecture ksize=round(6*sigma)"""
    ksize = round(6*sigma)
    if ksize % 2 == 0:
        ksize += 1
    kernel = [np.exp(-0.5*np.square(x/sigma)) for x in range(-(ksize//2), ksize//2+1)]
    kernel = np.vstack(kernel)
    kernel = kernel / kernel.sum()
    return kernel


def task4():
    print_task_name('4. 2D Filtering')
    img = cv.imread('bonn.png', cv.IMREAD_GRAYSCALE)
    display_imgs({'bonn': img})
    sigma = 2 * np.sqrt(2)
    # 4.a
    blur_cv = cv.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)

    # 4.b
    kernel_1d = get_kernel(sigma)
    kernel = kernel_1d @ kernel_1d.T
    blur_filter2d = cv.filter2D(img, -1, kernel)

    # 4.c
    blur_sepFilter2d = cv.sepFilter2D(img, -1, kernelX=kernel_1d, kernelY=kernel_1d)

    display_imgs({
        'cv.GaussianBlur': blur_cv,
        'cv.filter2D': blur_filter2d,
        'cv.sepFilter2D': blur_sepFilter2d
    })

    print('Maximum pixel-wise differences:')
    blur_cv_int16 = blur_cv.astype(np.int16)
    blur_filter2d_int16 = blur_filter2d.astype(np.int16)
    blur_sepFilter2d_int16 = blur_sepFilter2d.astype(np.int16)
    print(f'cv.GaussianBlur and cv.filter2D: {np.abs(blur_cv_int16-blur_filter2d_int16).max()}')
    print(
        f'cv.GaussianBlur and cv.sepFilter2D: {np.abs(blur_cv_int16-blur_sepFilter2d_int16).max()}')
    print(
        f'cv.filter2D and cv.sepFilter2D: {np.abs(blur_filter2d_int16-blur_sepFilter2d_int16).max()}')


# task4()

# ************************************************
# ********************TASK5***********************


def task5():
    print_task_name('5. Multiple Gaussian Filters')
    img = cv.imread('bonn.png', cv.IMREAD_GRAYSCALE)
    display_imgs({'bonn': img})
    sigma_a = 2
    sigma_b = 2 * np.sqrt(2)
    blur_a = cv.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma_a, sigmaY=sigma_a)
    blur_a = cv.GaussianBlur(blur_a, ksize=(0, 0), sigmaX=sigma_a, sigmaY=sigma_a)
    blur_b = cv.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma_b, sigmaY=sigma_b)
    display_imgs({
        'Blurred twice with sigma=2': blur_a,
        'Blurred once with sigma 2*sqrt(2)': blur_b
    })
    diff = np.abs(blur_a - blur_b)
    print(f'Maximum pixel difference: {diff.max()}')


# task5()

# ************************************************
# ********************TASK7***********************

def add_salt_n_pepper_noise(img: np.ndarray):
    width, height = img.shape
    for i in range(width):
        for j in range(height):
            should_noise = np.random.choice([False, True], p=[0.7, 0.3])
            if should_noise:
                noise = np.random.choice([0, 255])
                img[i][j] = noise


def task7():
    print_task_name('7. Denoising')
    img = cv.imread('bonn.png', cv.IMREAD_GRAYSCALE)
    noised_img = img.copy()
    add_salt_n_pepper_noise(noised_img)
    display_imgs({'salt and pepper noised': noised_img})

    result = {
        'gaussian': (None, float('inf')),
        'median': (None, float('inf')),
        'bilateral': (None, float('inf'))
    }
    for filter_size in [1, 3, 5, 7, 9]:
        gaussian_blur = cv.GaussianBlur(noised_img, ksize=(filter_size, filter_size), sigmaX=0)
        median_blur = cv.medianBlur(noised_img, ksize=filter_size)
        bilateral_blur = cv.bilateralFilter(noised_img, d=filter_size, sigmaColor=75, sigmaSpace=75)

        gaussian_score = abs(img.mean() - gaussian_blur.mean())
        median_score = abs(img.mean() - median_blur.mean())
        bilateral_score = abs(img.mean() - bilateral_blur.mean())

        if gaussian_score < result['gaussian'][1]:
            result['gaussian'] = (gaussian_blur, gaussian_score)
        if median_score < result['median'][1]:
            result['median'] = (median_blur, median_score)
        if bilateral_score < result['bilateral'][1]:
            result['bilateral'] = (bilateral_blur, bilateral_score)

    display_imgs({
        'gaussian filter': result['gaussian'][0],
        'median filter': result['median'][0],
        'bilateral filter': result['bilateral'][0]
    })


task7()


# ************************************************
# ********************TASK8***********************


def task8():
    print_task_name('8. Separability of Filters')
    img = cv.imread('bonn.png', cv.IMREAD_GRAYSCALE)
    k1 = np.array([[0.0113, 0.0838, 0.0113],
                   [0.0838, 0.6193, 0.0838],
                   [0.0113, 0.0838, 0.0113]])
    k2 = np.array([[-0.8948, 0.1472, 1.1410],
                   [-1.9075, 0.1566, 2.1359],
                   [-0.8659, 0.0573, 1.0337]])

    filtered1 = cv.filter2D(img, -1, k1)
    filtered2 = cv.filter2D(img, -1, k2)

    display_imgs({
        'k1': filtered1,
        'k2': filtered2
    })
    w1, u1, vt1 = cv.SVDecomp(k1)
    w2, u2, vt2 = cv.SVDecomp(k2)

    filter1_approx = cv.sepFilter2D(img, -1, kernelX=w1[0]*u1[:, 0], kernelY=vt1[0, :])
    filter2_approx = cv.sepFilter2D(img, -1, kernelX=w2[0]*u2[:, 0], kernelY=vt2[0, :])

    display_imgs({
        'k1 approx': filter1_approx,
        'k2 approx': filter2_approx
    })


# task8()
