import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2


def plot_snake(ax, V, fill="green", line="red", alpha=1, with_txt=False):
    """plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0, :]).reshape((-1, 2))
    ax.plot(V_plt[:, 0], V_plt[:, 1], color=line, alpha=alpha)
    ax.scatter(
        V[:, 0], V[:, 1], color=fill, edgecolors="black", linewidth=2, s=50, alpha=alpha
    )
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))


def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 20  # number of points

    def u(i):
        return radius * np.cos(i) + w / 2

    def v(i):
        return radius * np.sin(i) + h / 2

    V = np.array([(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][:-1], "int32")
    return Im, V


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS


def compute_gradients(img, window_size: int = 3):

    img_blur = cv2.GaussianBlur(img, (0, 0), 5, borderType=cv2.BORDER_DEFAULT)
    grad_x = cv2.Sobel(img_blur, -1, dx=1, dy=0, ksize=window_size)
    grad_y = cv2.Sobel(img_blur, -1, dx=0, dy=1, ksize=window_size)

    grad_magnitude_2 = np.square(grad_x.astype(np.int32)) + np.square(
        grad_y.astype(np.int32)
    )

    return np.sqrt(grad_magnitude_2)


def neighbor_coordinates(center_loc: tuple, window_size: int, img_size: tuple):
    if window_size % 2 == 1:
        x, y = np.meshgrid(
            range(
                max(0, center_loc[1] - window_size // 2),
                min(img_size[0], center_loc[1] + window_size // 2 + 1),
            ),
            range(
                max(0, center_loc[0] - window_size // 2),
                min(img_size[1], center_loc[0] + window_size // 2 + 1),
            ),
        )
    else:
        raise RuntimeError("Window size must be an odd integer")
    return np.vstack((y.T.flatten(), x.T.flatten()))


def avg_dist(V: np.ndarray):
    avg_dist = np.sum(la.norm(V[:-1] - V[1:], 2, 1)) / (V.shape[0] - 1)
    return avg_dist


def viterbi_dp(states, unary_costs, avg_dist):
    alpha = 0.5
    beta = 1 - alpha

    # Store indices for backtracking
    indices = np.zeros((unary_costs.shape[0], unary_costs.shape[1] - 1), np.int8)

    # Store accumulated cost during the forward pass through the graph
    accumulated_cost = np.copy(unary_costs)
    # print(accumulated_cost)
    for i in range(states.shape[1] - 1):
        for j in range(states.shape[0]):
            binary_cost = np.abs(
                la.norm(states[:, i] - states[j, i + 1], 2, 1) - avg_dist
            )
            total_cost = (
                alpha * binary_cost
                + beta * accumulated_cost[:, i]
                + beta * accumulated_cost[j, i + 1]
            )
            indices[j, i] = np.argmin(total_cost)
            accumulated_cost[j, i + 1] = np.amin(total_cost)

    min_idx_last = np.argmin(accumulated_cost[:, -1])
    new_V = np.zeros((states.shape[1], states.shape[2]), np.int16)

    for i in range(new_V.shape[0] - 1, -1, -1):
        new_V[i] = states[min_idx_last, i]
        min_idx_last = indices[min_idx_last, i - 1]

    return new_V


def run(fpath, radius):
    """run experiment
    :param fpath:
    :param radius:
    :return:
    """
    Im, V = load_data(fpath, radius)

    # Create a loop back on first vertex
    V = np.vstack((V, V[0]))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 100
    window_size = 5

    grad_mag_2 = compute_gradients(Im)

    for t in range(n_steps):
        states = np.array(
            [neighbor_coordinates(V[i], window_size, Im.shape) for i in range(len(V))]
        ).transpose(2, 0, 1)

        if t <= 0.1 * n_steps:
            states[:, 0] = V[0]
            states[:, -1] = V[-1]
        else:
            states[:, 2] = V[2]

        unary_cost = -(grad_mag_2[states[:, :, 1], states[:, :, 0]])

        avg_d = avg_dist(V)

        V_new = viterbi_dp(states, unary_cost, avg_d)

        if np.sum(la.norm(V_new - V, 2, 1)) < 1e-2:
            break
        else:
            V = V_new

        ax.clear()
        ax.imshow(Im, cmap="gray")
        ax.set_title("frame " + str(t))
        plot_snake(ax, V, with_txt=True)
        plt.pause(0.01)

    plt.pause(2)


if __name__ == "__main__":
    run("images/ball.png", radius=120)
    run("images/coffee.png", radius=100)
