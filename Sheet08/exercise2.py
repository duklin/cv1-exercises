import cv2
import numpy as np
import matplotlib.pylab as plt

def nonMaxSuppression(F: np.ndarray, locs: np.ndarray, window_size: int = 3) -> (np.ndarray):
    """
    Performs non maximum suppression in the feature space

    Parameters
    ----------
    - F: numpy array with feature values of identified features in the image [M x N]
    - locs: location of features in the image [num_features, 2]
    - window_size: size for non maximum suppression window (default = 3)

    Returns
    -------
    - nonmax_F: feature locations after non maximum suppression
    """
    nonmax_F = np.zeros_like(locs)
    num_f = 0

    m, n = F.shape
    w = window_size // 2

    for f in range(locs.shape[0]):
        if F[locs[f, 0], locs[f, 1]] == np.amax(F[max(0, locs[f, 0] - w):min(locs[f, 0] + w + 1, m), max(0, locs[f, 1] - w):min(locs[f, 1] + w + 1, n)]):
            nonmax_F[num_f] = [locs[f, 0], locs[f, 1]]
            num_f += 1

    return nonmax_F[:num_f]


def main():
    # Load the image
    img_color = cv2.imread('data/exercise2/building.jpeg')
    img_gray = cv2.imread('data/exercise2/building.jpeg', 0).astype(np.float32)
    h, w = img_gray.shape
    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0, borderType=cv2.BORDER_DEFAULT)

    window_size = 5
    I_x = cv2.Sobel(img_blur, -1, dx=1, dy=0, ksize=window_size)
    I_y = cv2.Sobel(img_blur, -1, dx=0, dy=1, ksize=window_size)

    k_size = 5

    # Compute Structural Tensor
    M_00 = cv2.filter2D(np.square(I_x), -1, np.ones((k_size, k_size)) / (k_size ** 2), cv2.BORDER_CONSTANT)
    M_11 = cv2.filter2D(np.square(I_y), -1, np.ones((k_size, k_size)) / (k_size ** 2), cv2.BORDER_CONSTANT)
    M_01 = cv2.filter2D(np.multiply(I_x, I_y), -1, np.ones((k_size, k_size)) / (k_size ** 2), cv2.BORDER_CONSTANT)

    structural_tensor_shape = (2, 2, h, w)
    M = np.empty(structural_tensor_shape)
    M[0, 0] = M_00
    M[0, 1] = M_01
    M[1, 0] = M_01
    M[1, 1] = M_11

    determinant_M = np.multiply(M[0, 0], M[1, 1]) - np.multiply(M[0, 1], M[1, 0])
    trace_M = M[0, 0] + M[1, 1]

    # Harris Corner Detection
    response = determinant_M - 0.05 * np.square(trace_M)

    plt.title('Harris Corner Response')
    plt.imshow(response, cmap='Spectral')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.waitforbuttonpress(0)
    plt.close()

    response = np.multiply(response, response > 0)
    feature_locs = np.array(np.where(response != 0)).T
    corners = nonMaxSuppression(response, feature_locs, 5)

    plt.title('Harris Corners')
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.plot(corners[:, 1], corners[:, 0], 'ro', markersize=1)
    plt.xticks([])
    plt.yticks([])
    plt.waitforbuttonpress(0)
    plt.close()

    # Forstner Corner Detection
    thres_w = 1.7
    w = np.divide(determinant_M, trace_M)
    w = w >= thres_w * np.mean(w)
    plt.title('Forstner Corner Response (Size)')
    plt.imshow(w, cmap='gray')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.waitforbuttonpress(0)
    plt.close()

    thres_q = 0.8
    q = np.divide(4 * determinant_M, np.square(trace_M))
    q = q >= thres_q
    plt.title('Forstner Corner Response (Roundness)')
    plt.imshow(q, cmap='gray')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.waitforbuttonpress(0)
    plt.close()

    response = w & q
    corners = np.array(np.where(response != 0)).T

    plt.title('Forstner Corners')
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.plot(corners[:, 1], corners[:, 0], 'ro', markersize=1)
    plt.xticks([])
    plt.yticks([])
    plt.waitforbuttonpress(0)
    plt.close()

if __name__ == '__main__':
    main()
