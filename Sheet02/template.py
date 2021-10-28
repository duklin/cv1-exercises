import cv2
import numpy as np


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_convolution_using_fourier_transform(image, kernel):
    """Perform image convolution using the Fourier Transform and its inverse
    """
    img_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, image.shape[:2])
    img_blur_fft = kernel_fft * img_fft
    image_blur = np.fft.ifft2(img_blur_fft).real

    return image_blur.astype(np.uint8)


def task1():
    """Convolution in spatial domain can be computed by multiplication in the frequency domain
    """
    image = cv2.imread("./data/einstein.jpeg", 0)
    # Gives 1D kernel
    kernel_1d = cv2.getGaussianKernel(7, 1, cv2.CV_32F)
    # Generate 2D kernel
    kernel_2d = kernel_1d @ kernel_1d.T

    conv_result = cv2.filter2D(image, -1, kernel_2d, borderType=cv2.BORDER_CONSTANT)
    fft_result = get_convolution_using_fourier_transform(image, kernel_2d)

    cv2.imshow('Original', image)
    cv2.imshow('Conv', conv_result)
    cv2.imshow('FFT', fft_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Mean Absolute Difference
    abs_diff = cv2.absdiff(conv_result, fft_result)
    mean_abs_diff = np.mean(abs_diff)

    print("The mean absolute difference of the two blurred images is: {}".format(mean_abs_diff))


def normalized_cross_correlation(image, template):
    i_m, i_n = image.shape
    t_m, t_n = template.shape

    ncc_image = np.zeros((i_m, i_n))

    zero_mean_template = template - np.mean(template)

    for row in range(t_m // 2, i_m - t_m // 2):
        for col in range(t_n // 2, i_n - t_n // 2):
            image_patch = image[row - t_m // 2:row +
                                t_m // 2 + 1, col - t_n // 2:col + t_n // 2 + 1]
            zero_mean_image_patch = image_patch - np.mean(image_patch)
            ncc_image[row, col] = np.sum(np.multiply(zero_mean_template, zero_mean_image_patch)) / np.sqrt(
                np.sum(np.square(zero_mean_template)) * np.sum(np.square(zero_mean_image_patch)))

    return ncc_image


def task2():
    """Template matching using normalized cross-correlation as similarity measures
    """
    image = cv2.imread("./data/lena.png", 0)
    template = cv2.imread("./data/eye.png", 0)

    t_m, t_n = template.shape

    result_ncc = normalized_cross_correlation(image, template)

    # Threshold the result for a similarity greater than 0.7
    loc_r, loc_c = np.where(result_ncc >= 0.7)

    # Draw rectangles around the thresholded pixels representing the detected template
    # Top left pixel for bounding box of size equal to template size
    loc_r = loc_r - t_m // 2
    loc_c = loc_c - t_n // 2
    for i in range(loc_r.shape[0]):
        cv2.rectangle(image, (loc_r[i], loc_c[i]), (
                      loc_r[i] + t_n, loc_c[i] + t_m), (100), 1)

    display_image('Matched Template with bounding box', image)


def build_gaussian_pyramid_opencv(image, num_levels):
    pyramid = {'level_0': image}
    for i in range(num_levels - 1):
        pyramid['level_{}'.format(i + 1)] = cv2.pyrDown(pyramid['level_{}'.format(i)])

    return pyramid


def build_gaussian_pyramid(image, num_levels):
    pyramid = {'level_0': image}
    for i in range(num_levels - 1):
        blurred_img = cv2.GaussianBlur(pyramid['level_{}'.format(i)])


def template_matching_multiple_scales(pyramid, template):
    # TODO: implement
    raise NotImplementedError


def task3():
    image = cv2.imread("./data/traffic.jpg", 0)
    template = cv2.imread("./data/template.jpg", 0)

    cv_pyramid = build_gaussian_pyramid_opencv(image, 4)

    my_pyramid = build_gaussian_pyramid(image, 4)
    my_pyramid_template = build_gaussian_pyramid(template, 4)

    # TODO: compare and print mean absolute difference at each level

    # TODO: calculate the time needed for template matching without the pyramid

    result = template_matching_multiple_scales(my_pyramid, my_pyramid_template)
    # TODO: calculate the time needed for template matching with the pyramid

    # TODO: show the template matching results using the pyramid


def get_derivative_of_gaussian_kernel(size, sigma):
    # TODO: implement
    raise NotImplementedError


def task4():
    image = cv2.imread("../data/einstein.jpeg", 0)

    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

    edges_x = None  # TODO: convolve with kernel_x
    edges_y = None  # TODO: convolve with kernel_y

    magnitude = None  # TODO: compute edge magnitude
    direction = None  # TODO: compute edge direction

    cv2.imshow("Magnitude", magnitude)
    cv2.imshow("Direction", direction)


def l2_distance_transform_1D(edge_vector, positive_inf, negative_inf):
    l = len(edge_vector)
    k = 0

    v = np.zeros(l, np.int32)

    z = np.empty(l + 1)
    z[0] = negative_inf
    z[1] = positive_inf

    dist_tf = np.zeros(l)

    q = 1
    while q < l:
        s = ((edge_vector[q] + q ** 2) - (edge_vector[v[k]] + v[k] ** 2)) / (2 * (q - v[k]))
        if s <= z[k]:
            k = k - 1
            continue
        else:
            k = k + 1
            v[k] = q
            z[k] = s
            z[k + 1] = positive_inf
            q = q + 1
    k = 0
    for q in range(l):
        while z[k + 1] < q:
            k = k + 1
        dist_tf[q] = (q - v[k]) ** 2 + edge_vector[v[k]]

    return dist_tf


def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    dist_column_wise = np.zeros_like(edge_function, np.uint32)
    for col in range(edge_function.shape[1]):
        dist_column_wise[:, col] = l2_distance_transform_1D(edge_function[:, col], positive_inf, negative_inf)
    
    l2_dist_tf = np.zeros_like(edge_function, np.uint32)
    for row in range(edge_function.shape[0]):
        l2_dist_tf[row] = l2_distance_transform_1D(dist_column_wise[row], positive_inf, negative_inf)

    l2_dist_tf = l2_dist_tf * 255 / np.max(l2_dist_tf)
    return l2_dist_tf
    




def task5():
    image = cv2.imread("./data/traffic.jpg", 0)

    # Detect Edges
    edges = cv2.Canny(image, 200, 225)
    display_image('edges', edges)

    positive_inf = np.inf
    negative_inf = -np.inf

    binary_edges = np.where(edges == 255, 0, image.shape[0] ** 2 + image.shape[1] ** 2)
    dist_transfom_mine = l2_distance_transform_2D(binary_edges, positive_inf, negative_inf)
    display_image('dist_mine', dist_transfom_mine)
    
    _, binary_edges = cv2.threshold(edges, 128, 255, type=cv2.THRESH_BINARY_INV)
    dist_transfom_cv = cv2.distanceTransform(binary_edges, cv2.DIST_L2, 5, dstType=cv2.CV_32F)
    display_image('dist_cv', dist_transfom_cv)

    # Mean Absolute Difference
    abs_diff = np.abs(dist_transfom_mine - dist_transfom_cv)
    mean_abs_diff = np.mean(abs_diff)

    print("The mean absolute difference of the two distance transforms is: {}".format(mean_abs_diff))


if __name__ == "__main__":
    # task1()
    # task2()
    # task3()
    # task4()
    task5()