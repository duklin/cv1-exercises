import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_data():
    """loads the data for this task
    :return:
    """
    fpath = "images/ball.png"
    radius = 70
    Im = cv2.imread(fpath, 0).astype("float32") / 255  # 0 .. 1

    # we resize the image to speed-up the level set method
    Im = cv2.resize(Im, dsize=(0, 0), fx=0.5, fy=0.5)

    height, width = Im.shape

    centre = (width // 2, height // 2)
    Y, X = np.ogrid[:height, :width]
    phi = radius - np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return Im, phi


def get_contour(phi):
    """get all points on the contour
    :param phi:
    :return: [(x, y), (x, y), ....]  points on contour
    """
    eps = 1
    A = (phi > -eps) * 1
    B = (phi < eps) * 1
    D = (A - B).astype(np.int32)
    D = (D == 0) * 1
    Y, X = np.nonzero(D)
    return np.array([X, Y]).transpose()


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here


def get_phi_derivatives(phi: np.ndarray) -> dict:
    """Returns first and second derivatives of the level-set function
    approximated with central difference"""
    phi = np.pad(phi, pad_width=1)
    phi_x = 0.5 * (phi[1:-1, 2:] - phi[1:-1, :-2])
    phi_y = 0.5 * (phi[2:, 1:-1] - phi[:-2, 1:-1])
    phi_xx = phi[1:-1, 2:] - 2 * phi[1:-1, 1:-1] + phi[1:-1, :-2]
    phi_yy = phi[2:, 1:-1] - 2 * phi[1:-1, 1:-1] + phi[:-2, 1:-1]
    phi_xy = 0.25 * (phi[2:, 2:] - phi[:-2, 2:] - phi[2:, :-2] + phi[:-2, :-2])
    return {"x": phi_x, "y": phi_y, "xx": phi_xx, "yy": phi_yy, "xy": phi_xy}


def get_phi_onesided_derivatives(phi: np.ndarray) -> dict:
    """Returns first derivatives of the level-set function
    approximated with one-sided difference"""
    phi = np.pad(phi, pad_width=1)
    right = phi[1:-1, 2:] - phi[1:-1, 1:-1]
    left = phi[1:-1, 1:-1] - phi[1:-1, :-2]
    down = phi[2:, 1:-1] - phi[1:-1, 1:-1]
    up = phi[1:-1, 1:-1] - phi[:-2, 1:-1]
    return {"right": right, "left": left, "up": up, "down": down}


def get_gradient(img: np.ndarray) -> np.ndarray:
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(np.square(sobelx) + np.square(sobely))
    return grad


def get_weight_derivatives(w: np.ndarray) -> dict:
    """Returns first derivatives of the weighting function"""
    w = np.pad(w, pad_width=1)
    w_x = 0.5 * (w[1:-1, 2:] - w[1:-1, :-2])
    w_y = 0.5 * (w[2:, 1:-1] - w[:-2, 1:-1])
    return {"x": w_x, "y": w_y}


if __name__ == "__main__":

    n_steps = 20000
    plot_every_n_step = 100

    Im, phi = load_data()

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ------------------------
    # your implementation here
    gradient = get_gradient(Im)
    w = 1 / (gradient + 1)
    dw = get_weight_derivatives(w)

    tau = 1 / (5 * np.max(w))
    epsilon = 1e-4
    # ------------------------

    for t in range(n_steps):

        # ------------------------
        # your implementation here
        d_phi = get_phi_derivatives(phi)
        mean_curv_term = (
            tau
            * w
            * (
                d_phi["xx"] * d_phi["y"] * d_phi["y"]
                - 2 * d_phi["x"] * d_phi["y"] * d_phi["xy"]
                + d_phi["yy"] * d_phi["x"] * d_phi["x"]
            )
            / (d_phi["x"] * d_phi["x"] + d_phi["y"] * d_phi["y"] + epsilon)
        )
        onesided_d_phi = get_phi_onesided_derivatives(phi)
        towards_edges_term = (
            np.max(dw["x"], 0) * onesided_d_phi["right"]
            + np.min(dw["x"], 0) * onesided_d_phi["left"]
            + np.max(dw["y"], 0) * onesided_d_phi["down"]
            + np.min(dw["y"], 0) * onesided_d_phi["up"]
        )
        phi = phi + mean_curv_term
        # phi = phi + mean_curv_term + towards_edges_term
        # ------------------------

        if t % plot_every_n_step == 0:
            ax1.clear()
            ax1.imshow(Im, cmap="gray")
            ax1.set_title("frame " + str(t))

            contour = get_contour(phi)
            if len(contour) > 0:
                ax1.scatter(contour[:, 0], contour[:, 1], color="red", s=1)

            ax2.clear()
            ax2.imshow(phi)
            ax2.set_title(r"$\phi$", fontsize=22)
            plt.pause(0.01)

    plt.show()
