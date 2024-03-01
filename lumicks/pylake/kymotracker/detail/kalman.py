import numpy as np

from dataclasses import dataclass


@dataclass(slots=True)
class PredictionError:
    m: float = 0
    n: int = 0
    M: float = 0

    def with_update(self, kalman_update):
        """Update the average prediction error"""

        # Equation 3.19

        # The kalman update is defined by the kalman gain times the innovation
        #
        #   x_updated = x_predicition + kalman_update = x_prediction + K * (z - H * x)
        #
        # Here z is the measurement, K is the Kalman gain and x is the state
        #
        # kalman update is x_hat = x_prediction + K * (z - H * x_prediction)
        n = self.n + 1
        m = self.m + (kalman_update - self.m) / n
        return PredictionError(m, n, self.M + (kalman_update - m).T * (kalman_update - m))

    def process_noise(self):
        if self.n == 0:
            return 1.0

        return self.M / self.n


@dataclass(slots=True)
class FilterState:
    state: np.ndarray
    cov: np.ndarray

    # Serves as an estimate of the process noise. We do an online estimate
    # of this.
    prediction_error: PredictionError = PredictionError()

    def with_state(self, state, cov):
        return FilterState(state, cov, self.prediction_error)

    def update_process_error(self, state, cov, kalman_update):
        return FilterState(state, cov, self.prediction_error.with_update(kalman_update))


class KalmanFilter:
    def __init__(
        self,
        transition_matrix,
        observation_matrix,
        # process_noise=None,
        measurement_noise=None,
    ):
        self.dim = transition_matrix.shape[0]
        self.transition_matrix = transition_matrix
        self.observation_matrix = observation_matrix
        # self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Check matrix consistency
        assert self.transition_matrix.shape[1] == self.dim
        assert self.observation_matrix.shape[1] == self.dim
        assert self.measurement_noise.shape == (self.observation_matrix.shape[0],) * 2
        # assert self.process_noise.shape == (self.dim, self.dim)

    def timestep(self, state):
        """Predict a state update"""
        return state.with_state(
            self.transition_matrix @ state.state,
            self.transition_matrix @ state.cov @ self.transition_matrix.T
            + state.prediction_error.process_noise(),
        )

    def inverse_system_uncertainty(self, state):
        cov_observed = state.cov @ self.observation_matrix.T
        system_uncertainty = self.observation_matrix @ cov_observed + self.measurement_noise
        return np.linalg.pinv(system_uncertainty)

    def add_measurement(self, state, measurement):
        """Perform a Kalman update step"""
        cov_observed = state.cov @ self.observation_matrix.T
        gain = cov_observed @ self.inverse_system_uncertainty(state)

        # State update is the gain times innovation
        state_update = gain @ (measurement - self.observation_matrix @ state.state)

        new_state = state.update_process_error(
            state.state + state_update,
            (np.eye(self.dim) - gain @ self.observation_matrix) @ state.cov,
            state_update,
        )

        return new_state

    def measurement_pdf(self, state):
        """Determines the measurement pdf based on the current state"""
        h = self.observation_matrix  # shorthand
        return h @ state.state, h @ state.cov @ h.T + self.measurement_noise

    def predict(self, state, measurement):
        """Determine measurement probability densities"""
        from scipy.stats import multivariate_normal

        meas, meas_cov = self.measurement_pdf(state)
        return multivariate_normal.pdf(measurement, mean=meas, cov=meas_cov)


def generate_constant_velocity_model(n_states=2, dt=1, observation_noise=0.1, n_obs=1):
    observation_matrix = np.zeros((n_obs, n_states))
    observation_matrix[0, 0] = 1
    return KalmanFilter(
        transition_matrix=np.array([[1.0, dt], [0.0, 1.0]]),
        observation_matrix=observation_matrix,
        measurement_noise=np.eye(n_obs) * observation_noise,
    )


def generate_diffusion_model(n_states=2, dt=1, observation_noise=0.1, n_obs=1):
    observation_matrix = np.zeros((n_obs, n_states))
    observation_matrix[0, 0] = 1
    return KalmanFilter(
        transition_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
        observation_matrix=observation_matrix,
        measurement_noise=np.eye(n_obs) * observation_noise,
    )
