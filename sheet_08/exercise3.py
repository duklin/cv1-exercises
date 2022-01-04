import cv2
import numpy as np
import matplotlib.pylab as plt

from sklearn.metrics import pairwise_distances

def compute_matches(D1: np.ndarray, D2: np.ndarray, thresh_1: float = 5, thresh_lowe: float = 0.5) -> (np.ndarray):
    """
    Computes matches for two images using the descriptors, use the Lowe's criterea to determine the best match.
    Parameters
    ----------
    - D1 : descriptors for image 1 corners [num_corners x 128]
    - D2 : descriptors for image 2 corners [num_corners x 128]
    - thresh_1: thresholf for the maximum match distance
    - thresh_lowe: threshold for Lowe's ratio test
 
    Returns
    ----------
    - M : [cornerIdx1, cornerIdx2] each row contains indices of corresponding keypoints [num_matches x 2]
    """
    T = 5 # Threshold for maximum error in a match

    l_d1 = D1.shape[0]
    l_d2 = D2.shape[0]

    M = np.zeros((min(l_d1, l_d2), 2), dtype=int)

    num_matches = 0
    # Compute Euclidean distance between each pair of descriptors in image 1 and image 2
    distances = pairwise_distances(D1, D2, 'euclidean', n_jobs=-1)

    for i in range(distances.shape[0]):
        sort_idx = np.argsort(distances[i])
        # Find two keypoints in image 2 with least descriptor distance from ith keypoint in image 1
        idx_min_1 = sort_idx[0]
        idx_min_2 = sort_idx[1]
        min_1 = distances[i, idx_min_1]
        min_2 = distances[i, idx_min_2]
        # Lowe's criteria 1
        if min_1 < thresh_1:
            # Lowe's criteria 2
            if ((min_1 / min_2) < thresh_lowe):
                M[num_matches] = np.array([int(i), int(idx_min_1)])
                num_matches += 1

    return M[:num_matches]


def plot_matches(I1: np.ndarray, I2: np.ndarray, C1: np.ndarray, C2: np.ndarray, M: np.ndarray) -> (None):
    """ 
    Plots the matches between the two images
    Parameters
    ----------
    - I1: Image 1
    - I2: Image 2
    - C1: Corners in image 1
    - C2: Corners in image 2
    - M: Matches between the corners in two images
    """
    # Create a new image with containing both images
    W = I1.shape[1] + I2.shape[1]
    H = np.max([I1.shape[0], I2.shape[0]])
    D = I1.shape[2]

    I_new = np.zeros((H, W, D), dtype=np.uint8)
    I_new[0:I1.shape[0], 0:I1.shape[1]] = I1
    I_new[0:I2.shape[0], I1.shape[1]:I1.shape[1] + I2.shape[1]] = I2

    # plot matches
    plt.imshow(I_new)
    for i in range(M.shape[0]):
        p1 = C1[M[i, 0]].pt
        p2 = C2[M[i, 1]].pt + np.array([I1.shape[1], 0])
        plt.plot(p1[0], p1[1], 'rx')
        plt.plot(p2[0], p2[1], 'rx')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-y')
    plt.waitforbuttonpress(0)
    plt.close()

def main():
    # Load the images
    img_color_1 = cv2.cvtColor(cv2.imread('data/exercise3/mountain1.png'), cv2.COLOR_BGR2RGB)
    img_gray_1 = cv2.imread('data/exercise3/mountain1.png', 0)
    img_color_2 = cv2.cvtColor(cv2.imread('data/exercise3/mountain2.png'), cv2.COLOR_BGR2RGB)
    img_gray_2 = cv2.imread('data/exercise3/mountain2.png', 0)

    # extract sift keypoints and descriptors
    sift = cv2.SIFT_create()

    keypts_1, descriptors_1 = sift.detectAndCompute(img_gray_1, None)
    keypts_2, descriptors_2 = sift.detectAndCompute(img_gray_2, None)

    # your own implementation of matching
    matches = compute_matches(descriptors_1, descriptors_2, 100, 0.4)

    # display the matches
    plot_matches(img_color_1, img_color_2, keypts_1, keypts_2, matches)
    

if __name__ == '__main__':
    main()
