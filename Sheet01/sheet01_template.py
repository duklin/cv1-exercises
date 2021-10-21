import cv2 as cv
import numpy as np
import random
import time


# ********************TASK1***********************
def integral_image(img):
    # Create a matrix of zeros adding a row and a column each to the original image dimensions
    int_img = np.zeros((img.shape[0] + 1, img.shape[1] + 1))

    # Iterate over each pixel in the image and recursively update the integral image
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            int_img[row + 1, col + 1] = int_img[row, col + 1] + \
                int_img[row + 1, col] - \
                int_img[row, col] + img[row, col]

    return int_img


def sum_image(image):
    sum_img_pixels = np.sum(image)
    return sum_img_pixels


def task1():
    print("[Task 1 Start .....]")
    img_bgr = cv.imread('bonn.png')
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    # Task 1a
    print("[Task 1a]")
    int_img_own = integral_image(img_gray)
    cv.imshow('Task 1 a: Integral Image Own Implementation',
              np.uint8(int_img_own[1:, 1:] * 255 / int_img_own[-1, -1]))
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Task 1b
    print("[Task 1b]")
    num_pixels = img_gray.shape[0] * img_gray.shape[1]

    # i : Summing up each pixel value
    img_mean_gray_sum = sum_image(img_gray) / num_pixels
    print("Mean Gray value of the image computed by summing each pixel value: {}".format(
        img_mean_gray_sum))

    # ii : Using integral image generated using OpenCV Function "integral"
    int_img_cv = cv.integral(img_gray)
    img_mean_gray_cv = int_img_cv[-1, -1] / num_pixels
    print("Mean Gray value of the image computed from the integral image using the opencv function: {}".format(
        img_mean_gray_cv))

    # iii :  Using integral image generated using own implementation for Integral Image
    img_mean_gray_own = int_img_own[-1, -1] / num_pixels
    print("Mean Gray value of the image computed from the integral image using own function: {}".format(
        img_mean_gray_own))

    # Task 1c
    print("[Task 1c]")
    # Select 10 coordinates at random denoting the top left pixel of the 100x100 image patches
    rows = np.random.randint(1, img_gray.shape[0] - 100, 10)
    cols = np.random.randint(1, img_gray.shape[1] - 100, 10)

    patch_size_x = 100
    patch_size_y = 100

    # i : Summing up each pixel value
    start_time = time.time()
    for n in range(10):
        img_mean_gray_sum = sum_image(
            img_gray[rows[n]:rows[n] + patch_size_y, cols[n]:cols[n] + patch_size_x]) / (patch_size_x * patch_size_y)
        # print("Mean Gray value of the random image patch {} computed by summing each pixel value: {}".format(
        #     n + 1, img_mean_gray_sum))
    end_time = time.time()
    print("Run-time of the task using the 'summing each pixel' method is {} seconds\n".format(end_time - start_time))

    # ii : Using integral image generated using OpenCV Function "integral"
    start_time = time.time()
    for n in range(10):
        img_mean_gray_cv = (int_img_cv[rows[n] + patch_size_y - 1, cols[n] + patch_size_x - 1] -
                            int_img_cv[rows[n] - 1, cols[n] + patch_size_x - 1] -
                            int_img_cv[rows[n] + patch_size_y - 1, cols[n] - 1] +
                            int_img_cv[rows[n] - 1, cols[n] - 1]) / (patch_size_x * patch_size_y)

        # print("Mean Gray value of the random image patch {} computed from the integral image using the opencv function: {}".format(
        #     n + 1, img_mean_gray_cv))
    end_time = time.time()
    print("Run-time of the task using the 'opencv integral function' method is {} seconds\n".format(end_time - start_time))

    # iii : Using Own implementation of Integral Image
    start_time = time.time()
    for n in range(10):
        img_mean_gray_own = (int_img_own[rows[n] + patch_size_y - 1, cols[n] + patch_size_x - 1] -
                             int_img_own[rows[n] - 1, cols[n] + patch_size_x - 1] -
                             int_img_own[rows[n] + patch_size_y - 1, cols[n] - 1] +
                             int_img_own[rows[n] - 1, cols[n] - 1]) / (patch_size_x * patch_size_y)
        # print("Mean Gray value of the random image patch {} computed from the integral image using own integral function: {}".format(
        #     n + 1, img_mean_gray_own))
    end_time = time.time()
    print("Run-time of the task using the 'own integral function' method is {} seconds".format(end_time - start_time))
    print("[Task 1 End .....]\n")
# ************************************************


# ********************TASK2***********************
def equalize_hist_image(img):
    # Compute image pixel's probability mass function and cumulative mass function
    img_pdf, bins = np.histogram(img, bins=256, density=True)
    img_cdf = np.cumsum(img_pdf)

    # Compute new pixel values for the equalized histogram image
    img_eq_hist = np.copy(img)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            img_eq_hist[row, col] = np.round(img_cdf[img[row, col]] * 255)

    return img_eq_hist


def task2():
    print("[Task 2 Start .....]")
    img_bgr = cv.imread('bonn.png')
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    print("[Task 2a]")
    img_eq_hist_cv = cv.equalizeHist(img_gray)
    cv.imshow(
        'Image with Equalized Histogram using opencv function "equalizeHist"', img_eq_hist_cv)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("[Task 2b]")
    img_eq_hist_own = equalize_hist_image(img_gray)
    cv.imshow(
        'Image with Equalized Histogram using own implementaion', img_eq_hist_own)
    cv.waitKey(0)
    cv.destroyAllWindows()

    abs_pix_diff = np.abs(img_eq_hist_cv.astype(
        np.int16) - img_eq_hist_own.astype(np.int16))
    print("The Maximum Absolute Pixel-wise Error between the two computations of equalized histogram image is: {}".format(np.amax(abs_pix_diff)))
    print("[Task 2 End .....]\n")
# ************************************************


# ********************TASK4***********************
def get_kernel(sigma):
    k_size = round(3 * sigma * 2)           # Find nearest integer acc to 3 sigma rule
    k_size = k_size + ((k_size + 1) % 2)    # Ensure that kernel size is odd number
    kernel = np.zeros((k_size, 1), np.float32)
    for i in range(k_size):
        kernel[i] = np.exp(-0.5 * (((i - (k_size // 2)) / sigma) ** 2))
    kernel = kernel / np.sum(kernel)    # Normalize to sum to 1

    return kernel


def task4():
    print("[Task 4 Start .....]")
    img_bgr = cv.imread('bonn.png')
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    cv.imshow("Grayscale Image", img_gray)
    cv.waitKey(0)
    cv.destroyAllWindows()

    sigma = 2 * np.sqrt(2)

    print("[Task 4a]")
    gauss_blur_img_a = cv.GaussianBlur(
        img_gray, (0, 0), sigma, borderType=cv.BORDER_DEFAULT)
    cv.imshow("Gaussian Blur Image: cv.GaussianBlur", gauss_blur_img_a)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("[Task 4b]")
    gauss_blur_img_b = cv.filter2D(
        img_gray, -1, get_kernel(sigma) @ get_kernel(sigma).T, borderType=cv.BORDER_DEFAULT)
    cv.imshow("Gaussian Blur Image: cv.filter2D", gauss_blur_img_b)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("[Task 4c]")
    gauss_blur_img_c = cv.sepFilter2D(
        img_gray, -1, get_kernel(sigma), get_kernel(sigma), borderType=cv.BORDER_DEFAULT)
    cv.imshow("Gaussian Blur Image: cv.sepFilter2D", gauss_blur_img_c)
    cv.waitKey(0)
    cv.destroyAllWindows()

    diff_ab = np.abs(gauss_blur_img_a.astype(np.int16) -
                     gauss_blur_img_b.astype(np.int16))
    print("Maximum absolute pixel-wise difference for pair 4a and 4b is : {}".format(np.amax(diff_ab)))

    diff_bc = np.abs(gauss_blur_img_b.astype(np.int16) -
                     gauss_blur_img_c.astype(np.int16))
    print("Maximum absolute pixel-wise difference for pair 4b and 4c is : {}".format(np.amax(diff_bc)))

    diff_ca = np.abs(gauss_blur_img_c.astype(np.int16) -
                     gauss_blur_img_a.astype(np.int16))
    print("Maximum absolute pixel-wise difference for pair 4c and 4a is : {}".format(np.amax(diff_ca)))
    print("[Task 4 End .....]\n")
# ************************************************


# ********************TASK5***********************
def task5():
    print("[Task 5 Start .....]")
    img_bgr = cv.imread('bonn.png')
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    cv.imshow("Grayscale Image", img_gray)
    cv.waitKey(0)
    cv.destroyAllWindows()

    k_size = 5

    print("[Task 5a]")
    sigma_a = 2
    gauss_blur_img_a = cv.GaussianBlur(
        img_gray, (k_size, k_size), sigma_a, borderType=cv.BORDER_DEFAULT)          # First
    gauss_blur_img_a = cv.GaussianBlur(
        gauss_blur_img_a, (k_size, k_size), sigma_a, borderType=cv.BORDER_DEFAULT)  # Second
    cv.imshow("Gaussian Blur Image: twice with sigma = 2", gauss_blur_img_a)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("[Task 5b]")
    sigma_b = 2 * np.sqrt(2)
    gauss_blur_img_b = cv.GaussianBlur(
        img_gray, (k_size, k_size), sigma_b, borderType=cv.BORDER_DEFAULT)
    cv.imshow("Gaussian Blur Image: once with sigma = 2*sqrt(2)",
              gauss_blur_img_b)
    cv.waitKey(0)
    cv.destroyAllWindows()

    diff_ab = np.abs(gauss_blur_img_a.astype(np.int16) -
                     gauss_blur_img_b.astype(np.int16))
    print("Maximum absolute pixel-wise difference between 5a and 5b is : {}".format(np.amax(diff_ab)))
    print("[Task 5 End .....]\n")
# ************************************************


# ********************TASK7***********************
def add_salt_n_pepper_noise(img):
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # Choose to add salt and pepper noise with 30% chance
            if random.choices([True, False], [0.3, 0.7])[0]:
                # Randomly select white or black pixel as noise
                img[row, col] = random.choice([0, 255])
    return img


def task7():
    print("[Task 7 Start .....]")
    img_bgr = cv.imread('bonn.png')
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    img_noisy = add_salt_n_pepper_noise(img_gray)
    cv.imshow("Grayscale Image with salt and pepper noise", img_noisy)
    cv.waitKey(0)
    cv.destroyAllWindows()

    k_sizes = [3, 5, 7, 9]

    print("[Task 7a]")
    mean_dist = np.zeros(4)
    for k in range(4):
        gauss_denoise_img = cv.GaussianBlur(
            img_noisy, (k_sizes[k], k_sizes[k]), 0, 0, borderType=cv.BORDER_DEFAULT)
        mean_dist[k] = np.sqrt(np.sum(np.square(gauss_denoise_img.astype(np.int16) - img_gray.astype(np.int16))))
    k_min = np.argmin(mean_dist)
    print(k_sizes[k_min])
    gauss_denoise_img = cv.GaussianBlur(
        img_noisy, (k_sizes[k_min], k_sizes[k_min]), 0, 0, borderType=cv.BORDER_DEFAULT)
    cv.imshow("Denoised with Gaussian Kernel", gauss_denoise_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    mean_dist = np.zeros(4)
    print("[Task 7b]")
    for k in range(4):
        median_denoise_img = cv.medianBlur(img_noisy, k_sizes[k])
        mean_dist[k] = np.sqrt(np.sum(np.square(median_denoise_img.astype(np.int16) - img_gray.astype(np.int16))))
    k_min = np.argmin(mean_dist)
    print(k_sizes[k_min])
    median_denoise_img = cv.medianBlur(img_noisy, k_sizes[k_min])
    cv.imshow("Denoised with Median Filter", median_denoise_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    mean_dist = np.zeros(4)
    print("[Task 7c]")
    for k in range(4):
        bilateral_denoise_img = cv.bilateralFilter(img_noisy, k_sizes[k], 160, 160, borderType=cv.BORDER_DEFAULT)
        mean_dist[k] = np.sqrt(np.sum(np.square(bilateral_denoise_img.astype(np.int16) - img_gray.astype(np.int16))))
    k_min = np.argmin(mean_dist)
    print(k_sizes[k_min])
    bilateral_denoise_img = cv.bilateralFilter(img_noisy, k_sizes[k_min], 160, 160, borderType=cv.BORDER_DEFAULT)
    cv.imshow("Denoised with Bilateral Filter", bilateral_denoise_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("[Task 7 End .....]\n")
# ************************************************


# ********************TASK8***********************
def task8():
    print("[Task 8 Start .....]")
    img_bgr = cv.imread('bonn.png')
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    K1 = np.array([[0.0113, 0.0838, 0.0113], [0.0838, 0.6193, 0.0838], [0.0113, 0.0838, 0.0113]])
    K2 = np.array([[-0.8984, 0.1472, 1.1410], [-1.9075, 0.1566, 2.1359], [-0.8659, 0.0573, 1.0337]])

    print("[Task 8a]")
    img_a = cv.filter2D(img_gray, -1, K1, borderType=cv.BORDER_DEFAULT)
    img_a = cv.filter2D(img_a, -1, K2, borderType=cv.BORDER_DEFAULT)
    cv.imshow("Resulting image after applying the 2 given 2D Kernels", img_a)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("[Task 8b]")
    w1, u1, v1_t = cv.SVDecomp(K1)
    sqrt_sigma_1 = np.sqrt(w1[0, 0])
    img_b = cv.sepFilter2D(img_gray, -1, sqrt_sigma_1 * u1[:, 0], sqrt_sigma_1 * v1_t[0, :], borderType=cv.BORDER_DEFAULT)
    
    w2, u2, v2_t = cv.SVDecomp(K2)
    sqrt_sigma_2 = np.sqrt(w2[0, 0])
    img_b = cv.sepFilter2D(img_b, -1, sqrt_sigma_2 * u2[:, 0], sqrt_sigma_2 * v2_t[0, :], borderType=cv.BORDER_DEFAULT)

    cv.imshow("Resulting image after applying the 4 1D kernels obtained from SVD of the 2 given 2D Kernels", img_b)
    cv.waitKey(0)
    cv.destroyAllWindows()

    diff = np.abs(img_a.astype(np.int16) -
                     img_b.astype(np.int16))
    print("Maximum absolute pixel-wise difference between results of task 8a and 8b is : {}".format(np.amax(diff)))
    print("[Task 8 End .....]\n")

if __name__ == "__main__":
    # task1()
    # task2()
    # task4()
    # task5()
    task7()
    task8()