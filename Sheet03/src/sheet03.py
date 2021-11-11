import numpy as np
import cv2 as cv
import random

from numpy.lib.function_base import meshgrid


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
    clusters = [[] for i in range(k)]

    # initialize centers using some random points from data
    # ....

    convergence = False
    iterationNo = 0
    while not convergence:
        # assign each point to the cluster of closest center
        # ...

        # update clusters' centers and check for convergence
        # ...

        iterationNo += 1
        print('iterationNo = ', iterationNo)

    return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


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

    cost_ncut = np.sum(W[meshgrid(cluster_1, cluster_2)[0], meshgrid(cluster_1, cluster_2)[1]]) * ((1 / volume_1) + (1 / volume_2))
    print("Cost of the Normalized Graph Cut is: ", cost_ncut)

##############################################

if __name__ == "__main__":
    # task_1_a()
    # task_1_b()
    # task_2()
    # task_3_a()
    # task_3_b()
    # task_3_c()
    task_4_a()

