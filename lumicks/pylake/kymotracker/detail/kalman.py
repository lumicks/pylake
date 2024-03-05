from typing import List, Optional
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PredictionError:
    m: np.ndarray = np.array([[0, 0]])
    n: int = 0
    M: np.ndarray = np.array([[0, 0], [0, 0]])

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

        return PredictionError(m, n, self.M + (kalman_update - m).T @ (kalman_update - m))

    def process_noise(self):
        if self.n < 3:
            # TODO: Better fallback for absence of process noise estimate
            return np.eye(self.M.shape[0])

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

    def average_with(self, new_state):
        """Average the resulting state estimate with a new incoming one"""
        avg_cov = np.linalg.pinv(np.linalg.pinv(self.cov) + np.linalg.pinv(new_state.cov))
        weighted_mean = (
            np.linalg.pinv(self.cov) @ self.state + np.linalg.pinv(new_state.cov) @ new_state.state
        )
        return FilterState(
            avg_cov @ weighted_mean,
            avg_cov,
            new_state.prediction_error,
        )


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

    def reverse(self):
        return KalmanFilter(
            np.linalg.pinv(self.transition_matrix), self.observation_matrix, self.measurement_noise
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

    def predict(self, state, measurement, innovation_cutoff=None):
        """Determine measurement probability densities"""
        from scipy.stats import multivariate_normal

        meas, meas_cov = self.measurement_pdf(state)
        log_pdf = np.atleast_1d(multivariate_normal.logpdf(measurement, mean=meas, cov=meas_cov))
        if not innovation_cutoff:
            return log_pdf

        h = self.observation_matrix
        innovation_cov = (
            innovation_cutoff
            * np.sqrt(2 * h @ state.prediction_error.process_noise() @ h.T).squeeze()
        )
        pos = np.sqrt((h @ state.state - measurement) ** 2)
        log_pdf[pos > innovation_cov] = -np.inf

        return log_pdf


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


class Track:
    def __init__(self, coord, time, std):
        self.coordinate, self.time_idx, self.stdev = [], [], []
        self.prepend(coord, time, std)

    def prepend(self, coord, time, std):
        self.coordinate.insert(0, coord)
        self.time_idx.insert(0, time)
        self.stdev.insert(0, std)


@dataclass
class KalmanFrame:
    coordinates: np.ndarray
    time_points: np.ndarray
    source_point: np.ndarray  # Where were we coming from?
    log_pdf: np.ndarray  # Last log-pdf
    filter_states: List[List[FilterState]]
    motion_model: np.ndarray  # Which motion model are we using?
    track: List[Optional[Track]]

    @classmethod
    def _from_kymopeak_frame(cls, peaks, state=np.zeros((2, 1)), cov=np.eye(2), model_index=0):
        return cls(
            peaks.coordinates,
            peaks.time_points,
            np.full(peaks.coordinates.shape, -1),
            np.full(peaks.coordinates.shape, -np.inf),
            [[FilterState(np.array([pos, 0]), cov) for pos in peaks.coordinates]],
            np.zeros(peaks.coordinates.shape),
            [None for _ in range(len(peaks.coordinates))],
        )


def score_to_connections(score_matrix):
    if score_matrix.size == 0:
        return []

    connections, scores = [], []

    for _ in range(score_matrix.shape[0]):
        from_point, to_point = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        score = score_matrix[from_point, to_point]

        if np.isfinite(score):
            score_matrix[from_point, :] = -np.inf
            score_matrix[:, to_point] = -np.inf
            connections.append([from_point, to_point])
            scores.append(score)
        else:
            break

    return connections, scores


def stitch_from_end(forward):
    total = len(forward)

    all_tracks = []
    for frame_idx, (frame, prev_frame) in enumerate(zip(reversed(forward), reversed(forward[:-1]))):
        for track, coordinate, filter_state, source_point in zip(
            frame.track, frame.coordinates, frame.filter_states[0], frame.source_point
        ):
            if not track:
                track = Track(coordinate, total - frame_idx, filter_state.cov[0, 0])
                all_tracks.append(track)
            else:
                track.prepend(coordinate, total - frame_idx, filter_state.cov[0, 0])

            if source_point > -1:
                prev_frame.track[source_point] = track

    return all_tracks


def forward_pass(kalman_frames, kalman_filter, cutoff):
    """Initialization pass (forward)

    Parameters
    ----------
    kalman_frames : List[KalmanFrame]
        Frame of detected peaks associated with filter states
    kalman_filter : KalmanFilter
        Kalman filter used
    cutoff : float
        Cutoff parameter for connecting two points
    """
    frames = [frame for frame in kalman_frames]
    for ix, (from_frame, to_frame) in enumerate(zip(frames[:-1], frames[1:])):
        score_matrix = np.atleast_2d(
            [
                np.atleast_1d(kalman_filter.predict(state, to_frame.coordinates, cutoff))
                for state in from_frame.filter_states[0]
            ]
        )

        # Find the most likely connections and run the Kalman filters
        for (from_point, to_point), loglik in zip(*score_to_connections(score_matrix)):
            state = from_frame.filter_states[0][from_point]
            state = kalman_filter.timestep(state)
            state = kalman_filter.add_measurement(state, to_frame.coordinates[to_point])
            to_frame.filter_states[0][to_point] = state
            to_frame.source_point[to_point] = from_point
            to_frame.log_pdf[to_point] = loglik

    return frames


@dataclass
class EmptyFrame:
    coordinates: np.ndarray = np.atleast_1d([])
    source_point: np.ndarray = np.atleast_1d([])


def reverse_frames(frames):
    """Reverses the way the peaks are linked.

    Every peak contains a source which reflects where it came from. Since we want
    to run the filter in backwards fashion, we reverse where this information is stored
    so that the algorithm itself can stay the same for all passes.

    Instead of storing the linkage on the frame we move towards, we store it in the
    frame we came from. This also means moving the log-likelihood with.
    """
    # Reverse the linkages
    flipped_source_points = []
    flipped_log_pdf = []
    for ix, (from_frame, to_frame) in enumerate(zip(frames[:-1], frames[1:])):
        from_source_point = np.full(from_frame.coordinates.shape, -1)
        from_log_pdf = np.full(from_frame.coordinates.shape, -np.inf)
        for target_idx, (from_idx, log_pdf) in enumerate(
            zip(to_frame.source_point, to_frame.log_pdf)
        ):
            if from_idx > -1:
                from_source_point[from_idx] = target_idx
                from_log_pdf[from_idx] = log_pdf

        flipped_source_points.append(from_source_point)
        flipped_log_pdf.append(from_log_pdf)

    for frame, new_source_points, new_log_pdf in zip(
        frames, flipped_source_points, flipped_log_pdf
    ):
        frame.source_point = new_source_points
        frame.log_pdf = new_log_pdf

    frames[-1].source_point = np.full(frames[-1].coordinates.shape, -1)

    return [f for f in reversed(frames)]


def iterate_filters(kalman_frames, kalman_filter, cutoff):
    """Iterated Kalman Tracking pass.

    Parameters
    ----------
    kalman_frames : List[KalmanFrame]
        Frame of detected peaks associated with filter states
    kalman_filter : KalmanFilter
        Kalman filter used
    cutoff : float
        Cutoff parameter for connecting two points
    """
    frames = reverse_frames(kalman_frames)

    new_source_points = []
    for ix, (from_frame, to_frame) in enumerate(zip(frames[:-1], frames[1:])):
        if to_frame.coordinates.size == 0:
            new_source_points.append([])
            continue

        score_matrix = np.atleast_2d(
            [
                np.atleast_1d(kalman_filter.predict(state, to_frame.coordinates, cutoff))
                for state in from_frame.filter_states[0]
            ]
        )

        # Find the most likely connections and run the Kalman filters
        for (from_point, to_point), loglik in zip(*score_to_connections(score_matrix)):
            state = from_frame.filter_states[0][from_point]
            state = kalman_filter.timestep(state)

            # Update step (don't commit yet!)
            new_to_state = kalman_filter.add_measurement(state, to_frame.coordinates[to_point])

            # Are we finding the same connection as before?
            if to_frame.source_point[to_point] == from_point:
                # From here, we have two options, if the point we are considering
                # (to_frame->to_point) is the one we would've linked from in the previous pass and
                # the motion type is the same (TO-DO), then we're good and we can average these
                # two state estimates!
                old_to_state = to_frame.filter_states[0][to_point]
                to_frame.filter_states[0][to_point] = old_to_state.average_with(new_to_state)
                to_frame.source_point[to_point] = from_point

                if abs(to_frame.log_pdf[to_point] - loglik):
                    assert np.isfinite(to_frame.log_pdf[to_point])

                to_frame.log_pdf[to_point] = loglik  # Update log likelihood of the measurement

            else:
                # Let's see if the connection we are proposing is better.
                previous_connection_log_likelihood = to_frame.log_pdf[to_point]

                # TODO: Mark as new motion regime once we have multiple filters
                # From the paper: Indeed even though two consecutive modes can be the same, the
                # selection of two different measurements justifies the definition of two
                # independent motion regimes

                # We grab the one with the best likelihood
                if loglik > previous_connection_log_likelihood:
                    to_frame.filter_states[0][to_point] = new_to_state
                    to_frame.source_point[to_point] = from_point
                    to_frame.log_pdf[to_point] = loglik

    return frames
