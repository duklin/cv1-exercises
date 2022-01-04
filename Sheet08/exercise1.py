import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import cv2
import sklearn
import random
import matplotlib.pylab as plt

def check_face(img_path, k_eig_vec, mean_face, h, w, threshold):
    img = cv2.imread(img_path, 0).astype(np.float32)
    resized_img = cv2.resize(img, (h, w), interpolation = cv2.INTER_AREA).reshape(-1, 1)

    centered_img = resized_img - mean_face
    coeffs = k_eig_vec.T @ centered_img

    img_recon = mean_face + k_eig_vec @ coeffs

    # plt.imshow(img_recon.reshape(h, w), cmap='gray')
    # plt.waitforbuttonpress(0)
    # plt.close()

    error = np.linalg.norm(resized_img.astype(np.float32) - img_recon.astype(np.float32), 2)
    # print('Reconstruction Error: {}'.format(error))

    if error < threshold:
        face_flag = True
    else:
        face_flag = False

    return face_flag    


def main():
    random.seed(0)
    np.random.seed(0)

    # Loading the LFW dataset
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw.images.shape
    X = lfw.data
    n_pixels = X.shape[1]
    y = lfw.target  # y is the id of the person in the image

    # splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    
    k = 100

    # Center the Data
    mean_face = np.mean(X_train, 0).reshape(-1, 1)
    centered_X = X_train - mean_face.T

    # Compute the PCA 
    eig_val, eig_vec = np.linalg.eigh(centered_X.T @ centered_X)

    # Visualize Eigen Faces
    fig, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            n = i * 5 + j
            ax[i, j].axis('off')
            ax[i, j].set_title("{} $^t$$^h$\nEigenface".format(n+1))
            ax[i, j].imshow(eig_vec[:, -n-1].reshape(h, w), cmap='gray')
    plt.waitforbuttonpress(0)
    plt.close()

    # Compute reconstruction error
    img_trump = cv2.imread("data/exercise1/detect/face/trump.jpg", 0).astype(np.float32)

    resized_img = cv2.resize(img_trump, (h, w), interpolation = cv2.INTER_AREA).reshape(-1, 1)

    centered_img = resized_img - mean_face

    coeffs = eig_vec[:, -k:].T @ centered_img

    img_recon = mean_face + eig_vec[:, -k:] @ coeffs

    plt.imshow(img_recon.reshape(h, w), cmap='gray')
    plt.waitforbuttonpress(0)
    plt.close()

    error = np.linalg.norm(resized_img.astype(np.float32) - img_recon.astype(np.float32), 2)
    print('Task 1a: Reconstruction Error: {}'.format(error))


    TP = 0  # True Positives
    TN = 0  # True Negatives
    threshold = 1500

    # Perform face detection
    TP += check_face("data/exercise1/detect/face/boris.jpg", eig_vec[:, -k:], mean_face, h, w, threshold)
    TP += check_face("data/exercise1/detect/face/merkel.jpg", eig_vec[:, -k:], mean_face, h, w, threshold)
    TP += check_face("data/exercise1/detect/face/obama.jpg", eig_vec[:, -k:], mean_face, h, w, threshold)
    TP += check_face("data/exercise1/detect/face/putin.jpg", eig_vec[:, -k:], mean_face, h, w, threshold)
    TP += check_face("data/exercise1/detect/face/trump.jpg", eig_vec[:, -k:], mean_face, h, w, threshold)

    TN += not(check_face("data/exercise1/detect/other/cat.jpg", eig_vec[:, -k:], mean_face, h, w, threshold))
    TN += not(check_face("data/exercise1/detect/other/dog.jpg", eig_vec[:, -k:], mean_face, h, w, threshold))
    TN += not(check_face("data/exercise1/detect/other/flag.jpg", eig_vec[:, -k:], mean_face, h, w, threshold))
    TN += not(check_face("data/exercise1/detect/other/flower.jpg", eig_vec[:, -k:], mean_face, h, w, threshold))
    TN += not(check_face("data/exercise1/detect/other/monkey.jpg", eig_vec[:, -k:], mean_face, h, w, threshold))

    print("True positives (out of 5): ", TP)
    print("True negatives (out of 5): ", TN)

    print("Task 1b: Accuracy: ", (TP + TN) / 10)

    # Perform face recognition
    train_coeffs = eig_vec[:, -k:].T @ centered_X.T
    test_coeffs = eig_vec[:, -k:].T @ (X_test.T - mean_face)

    idx = []

    for test_coeff in test_coeffs.T:
        l2_dist = np.linalg.norm(test_coeff - train_coeffs.T, 1, 1)
        idx.append(np.argmin(l2_dist))
    
    predicted_y = y_train[idx].reshape(-1,)

    accuracy = np.sum((predicted_y - y_test) == 0) / len(y_test)

    print("Task 1c: Accuracy: ", accuracy)

if __name__ == '__main__':
    main()
