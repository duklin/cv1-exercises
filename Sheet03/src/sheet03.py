import numpy as np
import cv2 as cv
import random

##############################################
#     Task 1        ##########################
##############################################


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/shapes.png')
    '''
    ...
    your code ...
    ...
    '''


def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    accumulator = np.zeros((int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)))
    detected_lines = []
    '''
    ...
    your code ...
    ...
    '''
    return detected_lines, accumulator


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
    #detected_lines, accumulator = myHoughLines(edges, 1, 2, 50)
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 2        ##########################
##############################################


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
    theta_res = None # set the resolution of theta
    d_res = None # set the distance resolution
    #_, accumulator = myHoughLines(edges, d_res, theta_res, 50)
    '''
    ...
    your code ...
    ...
    '''

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
                print('Divide by zero encountered in updating cluster means...\n Reinitializing cluster center...\n')
                centers[j] = data[np.random.choice(data.shape[0], 1)]

        if np.sum(np.linalg.norm(centers - centers_old, 2, 1)) < 0.01:
            convergence = True
            break
        
        centers_old = np.copy(centers)
        iterationNo += 1
        print('iterationNo = ', iterationNo)

    return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png', 0)
    
    img_segmented = np.zeros((img.shape[0] * img.shape[1], 1))
    for k in [2, 4, 6]:
        index, centers = myKmeans(img.reshape(-1, 1), k)
        img_segmented = centers[index]
        cv.imshow('Image Segmented by Intensity: k = {}'.format(k), img_segmented.reshape(img.shape).astype(np.uint8))
        cv.waitKey(0)
        cv.destroyAllWindows()


def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    
    img_segmented = np.zeros((img.shape[0] * img.shape[1], 3))
    for k in [2, 4, 6]:
        index, centers = myKmeans(img.reshape(-1, 3), k)
        img_segmented = centers[index]
        cv.imshow('Image Segmented by Color: k = {}'.format(k), img_segmented.reshape((img.shape[0], img.shape[1], 3)).astype(np.uint8))
        cv.waitKey(0)
        cv.destroyAllWindows()


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png', 0)
    
    intensity_plus_pose = np.zeros((img.shape[0] * img.shape[1], 3))

    # First dimension of the feature space - Intensity values (0 to 255)
    intensity_plus_pose[:, 0] = img.flatten()

    # Second dimension of the feature space - y-coordinate of the image space scaled to (0 to 255)
    intensity_plus_pose[:, 1] = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))[0].T.flatten() * 255 / img.shape[0]

    # Third dimension of the feature space - x-coordinate of the image space scaled to (0 to 255)
    intensity_plus_pose[:, 2] = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))[1].T.flatten() * 255 / img.shape[1]

    for k in [2, 4, 6]:
        index, centers = myKmeans(intensity_plus_pose, k)
        img_segmented = centers[index]
        cv.imshow('Image Segmented by Intensity and Position of pixels: k = {}'.format(k), img_segmented[:, 0].reshape((img.shape[0], img.shape[1])).astype(np.uint8))
        cv.waitKey(0)
        cv.destroyAllWindows()


##############################################
#     Task 4        ##########################
##############################################

def task_4_a():
    print("Task 4 (a) ...")

    nodes = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])

    # Adjacency matrix
    W = np.array([[0, 1, 0.2, 1, 0, 0, 0, 0],
                  [1, 0, 0.1, 0, 1, 0, 0, 0],
                  [0.2, 0.1, 0, 1, 0, 1, 0.3, 0],
                  [1, 0, 1, 0, 0, 1, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 0, 0, 1, 0],
                  [0, 0, 0.3, 0, 1, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0, 1, 0]])
    # Degree Matrix
    D = np.diag(np.sum(W, 0))
    
    # Laplacian Matrix
    L = D - W

    _, eig_vec = np.linalg.eigh(L)

    y = eig_vec[:, 1]   # Eigen Vector corresponding to the second smallest Eigen value

    # Determine the clusters
    cluster_1 = np.where(y < 0)
    cluster_2 = np.where(y >= 0)
    print("Nodes in Cluster 1 are : ", nodes[cluster_1])
    print("Nodes in Cluster 2 are : ", nodes[cluster_2])

    # Compute volume of each cluster
    volume_1 = np.sum(D[cluster_1])
    volume_2 = np.sum(D[cluster_2])

    cost_ncut = np.sum(W[np.meshgrid(cluster_1, cluster_2)[0], np.meshgrid(cluster_1, cluster_2)[1]]) * ((1 / volume_1) + (1 / volume_2))
    print("Cost of the Normalized Graph Cut is: ", cost_ncut)

##############################################

if __name__ == "__main__":
    # task_1_a()
    # task_1_b()
    # task_2()
    task_3_a()
    task_3_b()
    task_3_c()
    task_4_a()