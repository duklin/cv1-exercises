import cv2
import numpy as np
import time


def print_task_name(task_name: str):
    """Helper function for making better terminal logging"""
    print(f"\n{task_name}\n{'-'*len(task_name)}")


def display_imgs(im_dict: dict):
    """Helper function for displaying images"""
    for window_name, img in im_dict.items():
        cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_convolution_using_fourier_transform(image, kernel):
    """Perform image convolution using the Fourier Transform and its inverse"""
    img_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, image.shape)
    img_blur_fft = kernel_fft * img_fft
    image_blur = np.fft.ifft2(img_blur_fft).real

    return image_blur.astype(np.uint8)


def task1():
    print_task_name("1. Fourier Transform")
    image = cv2.imread("./data/einstein.jpeg", 0)
    # Gives 1D kernel
    kernel_1d = cv2.getGaussianKernel(7, 1)
    # Generate 2D kernel
    kernel_2d = kernel_1d @ kernel_1d.T

    conv_result = cv2.filter2D(image, -1, kernel_2d)
    fft_result = get_convolution_using_fourier_transform(image, kernel_2d)

    display_imgs({
        'Original': image,
        'Convolution': conv_result,
        'FFT': fft_result
    })

    # Mean Absolute Difference
    abs_diff = cv2.absdiff(conv_result, fft_result)
    mean_abs_diff = np.mean(abs_diff)

    print(f"The mean absolute difference of the two blurred images is: {mean_abs_diff}")


def normalized_cross_correlation(image, template):
    """Return a normalized cross correlation image with
    same dimension as the provided `image` based on the
    provided `template`"""
    im_height, im_width = image.shape
    temp_height, temp_width = template.shape

    ncc_image = np.zeros((im_height, im_width))

    zero_mean_template = template - np.mean(template)

    if temp_height % 2 == 1:
        offset_y = 1
    else:
        offset_y = 0
    if temp_width % 2 == 1:
        offset_x = 1
    else:
        offset_x = 0

    for row in range(temp_height // 2, im_height - temp_height // 2):
        for col in range(temp_width // 2, im_width - temp_width // 2):
            patch = image[row-temp_height // 2: row + temp_height //
                          2 + offset_y, col - temp_width // 2:col + temp_width // 2 + offset_x]
            zero_mean_patch = patch - np.mean(patch)
            ncc_image[row, col] = np.sum(zero_mean_template*zero_mean_patch) / np.sqrt(
                np.sum(np.square(zero_mean_template)) * np.sum(np.square(zero_mean_patch)))

    return ncc_image


def template_match(image, template, thresh=0.7):
    """Return x and y coordinates in two numpy arrays
    of the center coordinates where the `template` is found
    in the `image` by using normalized cross correlation"""
    result_ncc = normalized_cross_correlation(image, template)
    match_rows, match_cols = np.where(result_ncc >= thresh)
    return match_rows, match_cols


def draw_bbox(image, center_x, center_y, width, height):
    """Draw in-place rectangle with dimensions `width` and `height`
    and center `(center_x, center_y)`"""
    cv2.rectangle(image, (center_x-width//2, center_y-height//2), (
        center_x+width//2, center_y+height//2), 128, 1)


def task2():
    print_task_name("2. Template Matching")
    image = cv2.imread("./data/lena.png", 0)
    template = cv2.imread("./data/eye.png", 0)

    temp_height, temp_width = template.shape

    match_rows, match_cols = template_match(image, template)

    # Draw rectangles around the thresholded pixels representing the detected template
    for match_row, match_col in zip(match_rows, match_cols):
        draw_bbox(image, match_col, match_row, temp_width, temp_height)

    display_imgs({'Matched Template with bounding box': image})


def build_gaussian_pyramid_opencv(image, num_levels):
    """Build Gaussian pyramid by using `cv2.pyrDown` method.
    Return a list of numpy arrays where the first element is the `image`
    and the last element is the image in the highest level
    of the pyramid"""
    pyramid = [image]
    for _ in range(num_levels-1):
        last_lvl = pyramid[-1]
        new_lvl = cv2.pyrDown(last_lvl)
        pyramid.append(new_lvl)
    return pyramid


def build_gaussian_pyramid(image, num_levels):
    """Build Gaussian pyramid by Gaussian filtering and subsampling.
    Return a list of numpy arrays where the first element is the `image`
    and the last element is the image in the highest level
    of the pyramid"""
    pyramid = [image]
    kernel = cv2.getGaussianKernel(ksize=5, sigma=0)
    for _ in range(num_levels-1):
        last_lvl = pyramid[-1]
        new_lvl = cv2.filter2D(last_lvl, -1, kernel)
        new_lvl = new_lvl[::2, ::2]
        pyramid.append(new_lvl)
    return pyramid


def template_matching_multiple_scales(image_pyramid, template_pyramid, thresh=0.8):
    """Use Gaussian Pyramids for image and template for faster template matching.
    Start with initial patch which is the whole image in the highest level.
    Then search only in patches with the following dimensions:
    (1.5*template height, 1.5*template width)
    Return arrays of x and y coordinates of the matches in the original image (level 0)
    """
    im_height, im_width = image_pyramid[-1].shape  # shape of the smallest image
    # set the smallest image as initial patch to look the template in
    patches_coords = [{
        'x': im_width//2,  # center x
        'y': im_height//2,  # center y
        'width': im_width,
        'height': im_height
    }]
    # iterate backwards
    for image, template in list(zip(image_pyramid, template_pyramid))[::-1]:
        # the list will contain the patches too look for in the lower level
        patches_coords_for_next_lvl = []
        temp_height, temp_width = template.shape

        for patch_coord in patches_coords:
            # check for boundaries
            y0 = max(patch_coord['y']-patch_coord['height']//2, 0)
            x0 = max(patch_coord['x']-patch_coord['width']//2, 0)
            y1 = min(patch_coord['y']+patch_coord['height']//2, image.shape[0])
            x1 = min(patch_coord['x']+patch_coord['width']//2, image.shape[1])
            img_patch = image[y0:y1, x0:x1]

            match_rows, match_cols = template_match(img_patch, template, thresh=thresh)
            # add (x0,y0) to match_rows and match_cols to convert them from 'patch' coordinates into 'image' coordinates
            # multiply the coordinates by two to convert them into coordinates for the lower level
            match_rows = (match_rows + y0) * 2
            match_cols = (match_cols + x0) * 2
            for match_row, match_col in zip(match_rows, match_cols):
                patches_coords_for_next_lvl.append({
                    'x': match_col,
                    'y': match_row,
                    'width': temp_width * 3,  # width of the patch will be 3/2
                    'height': temp_height * 3  # height of the patch will be 3/2
                })
        patches_coords = patches_coords_for_next_lvl

    # undo the multiplication on lines 166 and 167
    return match_rows//2, match_cols//2


def task3():
    print_task_name("3. Gaussian Pyramid")
    image = cv2.imread("./data/traffic.jpg", 0)
    template = cv2.imread("./data/traffic-template.png", 0)

    temp_height, temp_width = template.shape

    cv_pyramid = build_gaussian_pyramid_opencv(image, 4)

    my_pyramid = build_gaussian_pyramid(image, 4)
    my_pyramid_template = build_gaussian_pyramid(template, 4)

    for i, (cv_img, my_img) in enumerate(zip(cv_pyramid, my_pyramid)):
        diff = cv2.absdiff(cv_img, my_img)
        print(f'Mean difference at level {i} is {diff.mean():.2f}')

    start = time.perf_counter()
    match_rows, match_cols = template_match(image, template, thresh=0.85)
    end = time.perf_counter()
    print(f'Template matching without using the pyramid: {end-start:.2f}s.')

    image_bbox = np.copy(image)
    for match_row, match_col in zip(match_rows, match_cols):
        draw_bbox(image_bbox, match_col, match_row, temp_width, temp_height)
    display_imgs({'Template matching without using pyramid': image_bbox})

    start = time.perf_counter()
    match_rows, match_cols = template_matching_multiple_scales(
        my_pyramid, my_pyramid_template, thresh=0.85)
    end = time.perf_counter()
    print(f'Template matching when using the pyramid: {end-start:.2f}s.')

    image_bbox = np.copy(image)
    for match_row, match_col in zip(match_rows, match_cols):
        draw_bbox(image_bbox, match_col, match_row, temp_width, temp_height)

    display_imgs({'Template matching by using pyramid': image_bbox})


def get_derivative_of_gaussian_kernel(size: int, sigma: float):
    """Return a derivative of the Gaussian filter in x and y direction.
    The resulting filters will have `size-1, size-1` dimension"""
    kernel = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    kernel_deriv = np.diff(kernel, axis=0)
    deriv_size = kernel_deriv.shape[0]
    kernel_deriv_x = np.broadcast_to(kernel_deriv.T, (deriv_size, deriv_size))
    kernel_deriv_y = np.broadcast_to(kernel_deriv, (deriv_size, deriv_size))
    return kernel_deriv_x, kernel_deriv_y


def arr_to_img(array: np.ndarray) -> np.ndarray:
    """Linearly map an array into the range 0-255
    and convert it to `np.uint8` thus making it
    suitable for visualization"""
    slope = 255/(array.max()-array.min())
    array = np.round(slope*(array-array.min()))
    array = array.astype(np.uint8)
    return array


def task4():
    print_task_name("4. Edges")
    image = cv2.imread("./data/einstein.jpeg", 0)

    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

    edges_x = cv2.filter2D(image, cv2.CV_64F, kernel=kernel_x)
    edges_y = cv2.filter2D(image, cv2.CV_64F, kernel=kernel_y)

    magnitude = np.sqrt(np.square(edges_x)+np.square(edges_y))
    direction = np.arctan2(edges_y, edges_x)

    magnitude = arr_to_img(magnitude)
    direction = arr_to_img(direction)

    display_imgs({
        'magnitude': magnitude,
        'direction': direction
    })


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
        dist_column_wise[:, col] = l2_distance_transform_1D(
            edge_function[:, col], positive_inf, negative_inf)

    l2_dist_tf = np.zeros_like(edge_function, np.uint32)
    for row in range(edge_function.shape[0]):
        l2_dist_tf[row] = l2_distance_transform_1D(
            dist_column_wise[row], positive_inf, negative_inf)

    l2_dist_tf = l2_dist_tf * 255 / np.max(l2_dist_tf)
    return l2_dist_tf


def task5():
    print_task_name("5. Distance Transform")
    image = cv2.imread("./data/traffic.jpg", 0)

    # Detect Edges
    edges = cv2.Canny(image, 200, 225)
    display_imgs({'edges': edges})

    positive_inf = np.inf
    negative_inf = -np.inf

    binary_edges = np.where(edges == 255, 0, image.shape[0] ** 2 + image.shape[1] ** 2)
    dist_transfom_mine = l2_distance_transform_2D(binary_edges, positive_inf, negative_inf)
    display_imgs({'dist_mine': dist_transfom_mine})

    _, binary_edges = cv2.threshold(edges, 128, 255, type=cv2.THRESH_BINARY_INV)
    dist_transfom_cv = cv2.distanceTransform(binary_edges, cv2.DIST_L2, 5, dstType=cv2.CV_32F)
    display_imgs({'dist_cv': dist_transfom_cv})

    # Mean Absolute Difference
    abs_diff = np.abs(dist_transfom_mine - dist_transfom_cv)
    mean_abs_diff = np.mean(abs_diff)

    print("The mean absolute difference of the two distance transforms is: {}".format(mean_abs_diff))


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()
