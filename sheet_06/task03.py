import numpy as np
import cv2 as cv

"""
    load the image and foreground/background parts
    image: the original image
    background/foreground: numpy array of size (n_pixels, 3) (3 for RGB values),
    i.e. the data you need to train the GMM
"""


def read_image(filename):
    image = cv.imread(filename) / 255.0
    height, width = image.shape[:2]
    bounding_box = np.zeros(image.shape)
    bounding_box[90:350, 110:250, :] = 1
    bb_width, bb_height = 140, 260
    background = image[bounding_box == 0].reshape(
        (height * width - bb_width * bb_height, 3)
    )
    foreground = image[bounding_box == 1].reshape((bb_width * bb_height, 3))

    return image, foreground, background


class GMM(object):
    def __init__(self) -> None:
        super().__init__()
        self.model = []

    def gaussian_scores(self, data):
        num_components = len(self.model)
        n, m = data.shape
        scores = np.empty((num_components, n))

        for i, component in enumerate(self.model):
            lam = component["lambda"]
            mu = component["mu"]
            cov = component["cov"]
            cov_det = np.prod(cov)
            factor = 1 / (np.power(2 * np.pi, m / 2) * np.sqrt(cov_det))
            scores[i] = (
                lam
                * factor
                * np.exp(-0.5 * np.sum((1 / cov) * (data - mu) ** 2, axis=1))
            )

        return scores.T

    def fit_single_gaussian(self, data):
        """data is assumed to be in shape (n_samples, m_dimensions) for
        which we need to have m dimensional Gaussian distribution which
        is defined by m dimensional mean and mxm dimensional covariance matrix
        since we'll be using only the diagonal part of the cov matrix, we'll
        only return the diagonal which will be m dimensional array"""
        mu = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        self.model = [{"lambda": 1, "mu": mu, "cov": cov.diagonal()}]

    def estep(self, data):
        """Returns an array with shape (num_samples, num_components)"""
        rs = self.gaussian_scores(data)
        rs = rs / rs.sum(axis=1).reshape(-1, 1)
        return rs.T

    def mstep(self, data, rs):
        new_lambdas = np.sum(rs, axis=1) / np.sum(rs)
        for i, component in enumerate(self.model):
            r = rs[i].reshape(-1, 1)
            component["lambda"] = new_lambdas[i]
            mu = np.sum(r * data, axis=0) / np.sum(r)
            component["mu"] = mu
            component["cov"] = np.sum((r * (data - mu) ** 2), axis=0) / np.sum(r)

    def em_algorithm(self, data, n_iterations=10):
        for _ in range(n_iterations):
            rs = self.estep(data)
            self.mstep(data, rs)

    def split(self, epsilon=0.1):
        new_components = []
        for component in self.model:
            component["lambda"] /= 2
            mu_offset = epsilon * component["cov"]
            mu1 = component["mu"] + mu_offset
            mu2 = component["mu"] - mu_offset
            component["mu"] = mu1
            new_components.append(
                {"lambda": component["lambda"], "mu": mu2, "cov": component["cov"]}
            )
        self.model.extend(new_components)

    def probability(self, data):
        scores = self.gaussian_scores(data)
        scores = scores.sum(axis=1)
        scores = scores
        return scores

    def sample(self):
        p = [c["lambda"] for c in self.model]
        chosen = np.random.choice(np.arange(len(self.model)), size=1, p=p)
        return np.random.normal(loc=chosen["mu"], scale=np.sqrt(chosen["cov"]))

    def train(self, data, n_splits):
        self.fit_single_gaussian(data)
        for _ in range(n_splits):
            self.split()
        self.em_algorithm(data)


if __name__ == "__main__":

    image, foreground, background = read_image("data/person.jpg")

    """
    TODO: compute p(x|w=background) for each image pixel and manipulate the image such that
    everything below the threshold is black, display the resulting image
    Hint: Slide 64
    """

    gmm_background = GMM()
    gmm_background.train(background, 3)

    gmm_foreground = GMM()
    gmm_foreground.train(foreground, 3)

    scores_bg = gmm_background.probability(image.reshape(-1, 3))
    scores_fg = gmm_foreground.probability(image.reshape(-1, 3))

    mask = scores_bg / scores_fg
    mask = (mask - mask.mean()) / mask.std()

    mask = mask.reshape(image.shape[:2])
    threshold = -0.021452
    image[mask > threshold] = [0, 0, 0]

    cv.imshow("Background Substraction", image)
    cv.waitKey()
    cv.destroyAllWindows()
