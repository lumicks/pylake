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
    ):
        dim = len(initial_state)
        self.transition_matrix = transition_matrix
        self.observation_matrix = observation_matrix
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.inverse_system_uncertainty = np.full((dim, dim), np.inf)

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
        return h @ self.state, h @ self.cov * h.T + self.measurement_noise

    def mahalanobis(self, measurement):
        """Mahalanobis distance of measurement.

        Returns
        -------
        float
        """
        return np.sqrt(float(measurement.T @ self.inverse_system_uncertainty @ measurement))
