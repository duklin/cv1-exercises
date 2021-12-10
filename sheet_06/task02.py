import numpy as np
import matplotlib.pylab as plt

observations = np.load("data/observations.npy")


def get_observation(t):
    return observations[t]


class KalmanFilter(object):
    def __init__(self, psi, sigma_p, phi, sigma_m, tau):
        self.state = None
        self.covariance = None
        self.sigma_m = sigma_m
        self.tau = tau
        if tau == 0:
            self.psi = psi
            self.sigma_p = sigma_p
            self.phi = phi
        else:
            l = psi.shape[0]
            self.psi = np.zeros((l * (tau + 1), l * (tau + 1)))
            self.psi[:l, :l] = psi
            self.psi[l:, :-l] = np.eye(l * tau)

            self.sigma_p = np.zeros((l * (tau + 1), l * (tau + 1)))
            self.sigma_p[:l, :l] = sigma_p

            self.phi = np.zeros((phi.shape[0], l * (tau + 1)))
            self.phi[:, :l] = phi

    def init(self, init_state):
        l = len(init_state)
        if self.tau == 0:
            self.state = init_state
            self.covariance = np.eye(l)
        else:
            self.state = np.r_[np.zeros(l * self.tau), init_state]
            self.covariance = np.zeros((l * (self.tau + 1), l * (self.tau + 1)))
            self.covariance[-l:, -l:] = np.eye(l)

    def track(self, xt):
        state_pred = self.psi @ self.state
        state_cov_pred = self.sigma_p + self.psi @ self.covariance @ self.psi.T
        gain = (
            state_cov_pred
            @ self.phi.T
            @ np.linalg.inv(self.sigma_m + self.phi @ state_cov_pred @ self.phi.T)
        )

        self.state = state_pred + gain @ (xt - self.phi @ state_pred)
        self.covariance = (
            np.eye(self.state.shape[0]) - gain @ self.phi
        ) @ state_cov_pred

    def get_current_location(self):
        return self.state[-4:-2]


def perform_tracking(tracker):
    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        if tracker.tau != 0:
            if t > tracker.tau + 1:
                track.append(tracker.get_current_location())
        else:
            track.append(tracker.get_current_location())
    return track


def main():
    init_state = np.array([1, 0, 0, 0])

    psi = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    sp = 0.01
    sigma_p = np.array(
        [[sp, 0, 0, 0], [0, sp, 0, 0], [0, 0, sp * 4, 0], [0, 0, 0, sp * 4]]
    )

    phi = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    sm = 0.05
    sigma_m = np.array([[sm, 0], [0, sm]])

    tracker = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=0)
    tracker.init(init_state)

    fixed_lag_smoother = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=5)
    fixed_lag_smoother.init(init_state)

    track = perform_tracking(tracker)
    track_smoothed = perform_tracking(fixed_lag_smoother)

    plt.figure()
    plt.plot([x[0] for x in observations], [x[1] for x in observations])
    plt.plot([x[0] for x in track], [x[1] for x in track])
    plt.plot([x[0] for x in track_smoothed], [x[1] for x in track_smoothed])
    plt.legend(["Observations", "Tracker", "Smoothed tracker"])

    plt.show()


if __name__ == "__main__":
    main()
