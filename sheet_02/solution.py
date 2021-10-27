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


# task1()


def normalized_cross_correlation(image, template):
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
    result_ncc = normalized_cross_correlation(image, template)
    match_rows, match_cols = np.where(result_ncc >= thresh)
    return match_rows, match_cols


def draw_bbox(image, center_x, center_y, width, height):
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


# task2()


def build_gaussian_pyramid_opencv(image, num_levels):
    pyramid = [image]
    for _ in range(num_levels-1):
        last_lvl = pyramid[-1]
        new_lvl = cv2.pyrDown(last_lvl)
        pyramid.append(new_lvl)
    return pyramid


def build_gaussian_pyramid(image, num_levels):
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
    print(f'Template matching when using the pyramid: {end-start:.2f}')

    image_bbox = np.copy(image)
    for match_row, match_col in zip(match_rows, match_cols):
        draw_bbox(image_bbox, match_col, match_row, temp_width, temp_height)

    display_imgs({'Template matching by using pyramid': image_bbox})


# task3()


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


def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    # TODO: implement
    raise NotImplementedError


def task5():
    image = cv2.imread("../data/traffic.jpg", 0)

    edges = None  # TODO: compute edges

    dist_transfom_mine = l2_distance_transform_2D()
    dist_transfom_cv = None  # TODO: compute using opencv

    # TODO: compare and print mean absolute difference


# if __name__ == "__main__":
#     task1()
#     task2()
#     task3()
#     task4()
#     task5()
