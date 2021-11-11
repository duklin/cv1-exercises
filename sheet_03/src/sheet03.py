import numpy as np
import cv2 as cv


def display_imgs(im_dict: dict):
    """Helper function for displaying images"""
    for window_name, img in im_dict.items():
        cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


##############################################
#     Task 1        ##########################
##############################################


def draw_hough_lines(img: np.ndarray, lines: list):
    """Draw the lines defined with (rho, theta) pairs in `lines`"""
    for rho, theta in lines:
        cos = np.cos(theta)
        sin = np.sin(theta)
        x0 = rho * cos
        y0 = rho * sin
        x1 = int(x0 + 1000 * (-sin))
        y1 = int(y0 + 1000 * (cos))
        x2 = int(x0 - 1000 * (-sin))
        y2 = int(y0 - 1000 * (cos))
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread("../images/shapes.png")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(img_gray, 50, 150, apertureSize=3)
    detected_lines = cv.HoughLines(edges, 1, np.pi / 90, 50)
    detected_lines = np.squeeze(detected_lines)

    draw_hough_lines(img, detected_lines)

    display_imgs({"cv.HoughLine": img})


def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs, accumulator,
    accumulator axes values (distance and theta values)
    """
    accumulator = np.zeros(
        (int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)),
        dtype=np.int32,
    )
    detected_lines = []
    theta_vals = np.linspace(0, 180, num=accumulator.shape[0], endpoint=False)

    theta_vals = theta_vals * np.pi / 180
    thetas_cos = np.cos(theta_vals).reshape(-1, 1)
    thetas_sin = np.sin(theta_vals).reshape(-1, 1)

    rows, cols = img_edges.nonzero()
    rows = rows.reshape(-1, 1)
    cols = cols.reshape(-1, 1)

    # rhos is of dim (num of points, 180/theta_step_sz)
    rhos = cols * thetas_cos.T + rows * thetas_sin.T
    rho_vals = np.linspace(rhos.min(), rhos.max(), num=accumulator.shape[1])

    for rhos_per_point in rhos:
        for j, rho_per_point_per_theta in enumerate(rhos_per_point):
            # to which rho from rho_vals is closest
            rho_idx = np.abs(rho_per_point_per_theta - rho_vals).argmin()
            accumulator[j][rho_idx] += 1

    theta_idxs, rho_idxs = np.where(accumulator >= threshold)
    for theta_idx, rho_idx in zip(theta_idxs, rho_idxs):
        theta = theta_vals[theta_idx]
        rho = rho_vals[rho_idx]
        detected_lines.append((rho, theta))

    return detected_lines, accumulator, theta_vals, rho_vals


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread("../images/shapes.png")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img_gray, 50, 150, apertureSize=3)
    detected_lines, accumulator, _, _ = myHoughLines(edges, 1, 2, 50)

    draw_hough_lines(img, detected_lines)

    acc = cv.equalizeHist(accumulator.astype(np.uint8))

    display_imgs({"myHoughLines": img, "accumulator": acc})


##############################################
#     Task 2        ##########################
##############################################


def task_2():
    print("Task 2 ...")
    img = cv.imread("../images/line.png")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert the image into grayscale
    edges = cv.Canny(img_gray, 50, 150, apertureSize=3)  # detect the edges
    theta_res = 2  # set the resolution of theta
    d_res = 1  # set the distance resolution
    _, accumulator, theta_vals, rho_vals = myHoughLines(edges, d_res, theta_res, 50)

    acc_h, acc_w = accumulator.shape
    rows, cols = np.meshgrid(np.arange(acc_h), np.arange(acc_w), indexing="ij")

    # the idea is to transform the data from accumulator, and the axes values
    # (theta_vals and rho_vals) so that we have two 1d arrays theta_vals and rho_vals which
    # hold the theta and distance parameters for all the points in the Hough space
    # for better processing in the mean shift algorithm
    # every point is repeated as many times as the corresponding accumulator value
    theta_vals = theta_vals[rows]
    rho_vals = rho_vals[cols]
    theta_vals = theta_vals.repeat(accumulator.flatten())
    rho_vals = rho_vals.repeat(accumulator.flatten())

    # normalize the values for faster convergence
    theta_vals_normed = (theta_vals - theta_vals.mean()) / theta_vals.std()
    rho_vals_normed = (rho_vals - rho_vals.mean()) / rho_vals.std()

    # picking random (rho, theta) pair
    center_idx = np.random.randint(theta_vals.shape[0])
    center_theta = theta_vals_normed[center_idx]
    center_rho = rho_vals_normed[center_idx]

    # mean shift algorithm
    delta = 1e-5  # stopping criteria
    sigma = 0.2  # defining the neighborhood
    while True:
        kernel = np.exp(
            -(1 / sigma ** 2)
            * (
                np.square(theta_vals_normed - center_theta)
                + np.square(rho_vals_normed - center_rho)
            )
        )
        new_center_theta = np.sum(kernel * theta_vals_normed) / np.sum(kernel)
        new_center_rho = np.sum(kernel * rho_vals_normed) / np.sum(kernel)

        theta_delta = new_center_theta - center_theta
        rho_delta = new_center_rho - center_rho

        center_theta = new_center_theta
        center_rho = new_center_rho

        if rho_delta < delta and theta_delta < delta:
            break

    # unnormalize the final values
    center_theta = (center_theta * theta_vals.std()) + theta_vals.mean()
    center_rho = (center_rho * rho_vals.std()) + rho_vals.mean()

    draw_hough_lines(img, [(center_rho, center_theta)])
    display_imgs({"Mean Shift": img})


##############################################
#     Task 3        ##########################
##############################################


def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """
    centers = np.zeros((k, data.shape[1]))
    index = np.zeros(data.shape[0], dtype=int)

    # initialize centers using some random points from data
    centers_old = data[np.random.choice(data.shape[0], k, False)]

    convergence = False
    iterationNo = 0
    while not convergence:
        # assign each point to the cluster of closest center
        dist_to_centers = np.zeros((data.shape[0], k))
        for i in range(k):
            dist_to_centers[:, i] = np.linalg.norm(data - centers_old[i], 2, 1)
        index = np.argmin(dist_to_centers, 1)

        # update clusters' centers and check for convergence
        for j in range(k):
            if np.sum(index == j) != 0:
                centers[j] = np.sum(data[index == j], 0) / np.sum(index == j)
            else:
                print(
                    """Divide by zero encountered in updating cluster means...\n
                    Reinitializing cluster center...\n"""
                )
                centers[j] = data[np.random.choice(data.shape[0], 1)]

        if np.sum(np.linalg.norm(centers - centers_old, 2, 1)) < 0.01:
            convergence = True
            break

        centers_old = np.copy(centers)
        iterationNo += 1
        print("iterationNo = ", iterationNo)

    return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread("../images/flower.png", 0)

    img_segmented = np.zeros((img.shape[0] * img.shape[1], 1))
    for k in [2, 4, 6]:
        index, centers = myKmeans(img.reshape(-1, 1), k)
        img_segmented = centers[index]
        cv.imshow(
            "Image Segmented by Intensity: k = {}".format(k),
            img_segmented.reshape(img.shape).astype(np.uint8),
        )
        cv.waitKey(0)
        cv.destroyAllWindows()


def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread("../images/flower.png")

    img_segmented = np.zeros((img.shape[0] * img.shape[1], 3))
    for k in [2, 4, 6]:
        index, centers = myKmeans(img.reshape(-1, 3), k)
        img_segmented = centers[index]
        cv.imshow(
            "Image Segmented by Color: k = {}".format(k),
            img_segmented.reshape((img.shape[0], img.shape[1], 3)).astype(np.uint8),
        )
        cv.waitKey(0)
        cv.destroyAllWindows()


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread("../images/flower.png", 0)

    intensity_plus_pose = np.zeros((img.shape[0] * img.shape[1], 3))

    # First dimension of the feature space - Intensity values (0 to 255)
    intensity_plus_pose[:, 0] = img.flatten()

    # Second dimension of the feature space - y-coordinate of the image space scaled to (0 to 255)
    intensity_plus_pose[:, 1] = (
        np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))[0].T.flatten()
        * 255
        / img.shape[0]
    )

    # Third dimension of the feature space - x-coordinate of the image space scaled to (0 to 255)
    intensity_plus_pose[:, 2] = (
        np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))[1].T.flatten()
        * 255
        / img.shape[1]
    )

    for k in [2, 4, 6]:
        index, centers = myKmeans(intensity_plus_pose, k)
        img_segmented = centers[index]
        cv.imshow(
            "Image Segmented by Intensity and Position of pixels: k = {}".format(k),
            img_segmented[:, 0].reshape((img.shape[0], img.shape[1])).astype(np.uint8),
        )
        cv.waitKey(0)
        cv.destroyAllWindows()


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")

    nodes = np.array(["A", "B", "C", "D", "E", "F", "G", "H"])

    # Adjacency matrix
    W = np.array(
        [
            [0, 1, 0.2, 1, 0, 0, 0, 0],
            [1, 0, 0.1, 0, 1, 0, 0, 0],
            [0.2, 0.1, 0, 1, 0, 1, 0.3, 0],
            [1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 0.3, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 1, 0],
        ]
    )
    # Degree Matrix
    D = np.diag(np.sum(W, 0))

    # Laplacian Matrix
    L = D - W

    _, eig_vec = np.linalg.eigh(L)

    y = eig_vec[:, 1]  # Eigen Vector corresponding to the second smallest Eigen value

    # Determine the clusters
    cluster_1 = np.where(y < 0)
    cluster_2 = np.where(y >= 0)
    print("Nodes in Cluster 1 are : ", nodes[cluster_1])
    print("Nodes in Cluster 2 are : ", nodes[cluster_2])

    # Compute volume of each cluster
    volume_1 = np.sum(D[cluster_1])
    volume_2 = np.sum(D[cluster_2])

    cost_ncut = np.sum(
        W[np.meshgrid(cluster_1, cluster_2)[0], np.meshgrid(cluster_1, cluster_2)[1]]
    ) * ((1 / volume_1) + (1 / volume_2))
    print("Cost of the Normalized Graph Cut is: ", cost_ncut)


##############################################
##############################################
##############################################

if __name__ == "__main__":
    task_1_a()
    task_1_b()
    task_2()
    task_3_a()
    task_3_b()
    task_3_c()
    task_4_a()
