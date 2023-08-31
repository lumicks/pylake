import numpy as np


class KalmanFilter:
    def __init__(
        self,
        initial_state,
        initial_cov,
        transition_matrix,
        observation_matrix,
        process_noise=None,
        measurement_noise=None,
        validate=True,
    ):
        dim = len(initial_state)
        self.transition_matrix = transition_matrix
        self.observation_matrix = observation_matrix
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.inverse_system_uncertainty = np.full((dim, dim), np.inf)

        if validate:
            assert len(initial_state) == self.transition_matrix.shape[0]
            assert len(initial_state) == self.transition_matrix.shape[1]
            assert len(initial_state) == self.observation_matrix.shape[1]
            assert self.measurement_noise.shape == (self.observation_matrix.shape[0],) * 2
            assert self.process_noise.shape == (initial_state.shape[0],) * 2

        self.state = initial_state
        self.cov = initial_cov

    def timestep(self):
        """Predict a state update"""
        self.state = self.transition_matrix @ self.state
        self.cov = self.transition_matrix @ self.cov @ self.transition_matrix.T + self.process_noise

    def update_state(self, measurement):
        """Perform a Kalman update step"""
        cov_observed = self.cov @ self.observation_matrix.T
        system_uncertainty = self.observation_matrix @ cov_observed + self.measurement_noise
        self.inverse_system_uncertainty = np.linalg.pinv(system_uncertainty)

        gain = cov_observed @ self.inverse_system_uncertainty
        self.state = self.state + gain @ (measurement - self.observation_matrix @ self.state)
        self.cov = (np.eye(len(self.state)) - gain @ self.observation_matrix) @ self.cov

    @property
    def measurement_pdf(self):
        """Determines the measurement pdf based on the current state"""
        h = self.observation_matrix  # shorthand
        return h @ self.state, h @ self.cov @ h.T + self.measurement_noise

    def predict_measurement(self, measurement):
        """Determine measurement probability densities"""
        from scipy.stats import multivariate_normal

        meas, meas_cov = self.measurement_pdf
        return multivariate_normal.pdf(measurement, mean=meas, cov=meas_cov)

    def mahalanobis(self, measurement):
        """ "Mahalanobis distance of measurement.

        Returns
        -------
        mahalanobis : float
        """
        return np.sqrt(float(measurement.T @ self.inverse_system_uncertainty @ measurement))


def generate_constant_velocity_model(
    x0, cov, dt=1, process_noise=0.001, observation_noise=0.1, n_obs=1
):
    observation_matrix = np.zeros((n_obs, len(x0)))
    observation_matrix[0, 0] = 1
    return KalmanFilter(
        transition_matrix=np.array([[1.0, dt], [0.0, 1.0]]),
        observation_matrix=observation_matrix,
        process_noise=np.eye(len(x0)) * process_noise,
        measurement_noise=np.eye(n_obs) * observation_noise,
        initial_state=x0,
        initial_cov=cov,
    )


def generate_diffusion_model(x0, cov, process_noise=0.01, observation_noise=0.1, n_obs=1):
    observation_matrix = np.zeros((n_obs, len(x0)))
    observation_matrix[0, 0] = 1
    return KalmanFilter(
        transition_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
        observation_matrix=observation_matrix,
        process_noise=np.eye(len(x0)) * process_noise,
        measurement_noise=np.eye(n_obs) * observation_noise,
        initial_state=x0,
        initial_cov=cov,
    )
