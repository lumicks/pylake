import numpy as np


def kymo_diff_score(t, coordinate, t_prediction, vel, sigma, sigma_diffusion):
    """Estimate mean and sigma for future time points.

    This function returns a mean and sigma based on a velocity, fixed uncertainty and diffusion rate. This is used to
    compute a scoring function to determine whether two local maxima should be linked.

    We expect the future to be a combination of a base uncertainty, a diffusion and a constant velocity. The stdev for
    the particle at time point t for the diffusion process is given by:
        sigma(t) = N(mu(t), sigma*sqrt(t)).
    mu is simply given by the prediction based on constant velocity in this model.

    Parameters
    ----------
        t : float
            Current time
        coordinate : float
            Current position
        t_prediction : array_like
            Coordinates for which to compute a probability
        vel : float
            Estimated velocity of the particle
        sigma: float
            Starting uncertainty of the cone.
        sigma_diffusion : float
            Sigma representing diffusion. For diffusion, the spread is characterized by:
              Sigma(t) = sigma_diffusion * sqrt(t)
            Note that sigma_diffusion equates to sqrt(2*D) where D is the diffusion constant.
    """

    temporal_diff = t_prediction - t
    assert np.all(temporal_diff > 0)
    mu_t = coordinate + vel * temporal_diff
    sigma_t = sigma + sigma_diffusion * np.sqrt(temporal_diff)

    return mu_t, sigma_t


def build_score_matrix(lines, times, coordinates, model, sigma_cutoff):
    """Build score matrix for a combination of lines and positions.

    The score matrix contains a score for each line, point pair. For each line, we calculate a penalty function which
    reflects a score associated with connecting those two lines. In the current implementation these are based on a
    Gaussian cone around the most likely trajectory. Sigma provides a starting width, whereas diffusion widens the cone
    over time.

    Since the score is simply used as a relative scoring metric, and we are not interested in the absolute values,
    we do not have to exponentiate the Gaussian.

    Above a certain cutoff value (sigma_cutoff), connections are considered very unlikely and candidates in those
    regions are considered to not be able to be part of the same line.

    Parameters
    ----------
    lines: List[pylake.KymoLine]
    times: array_like
        Time points corresponding to the identified kymograph peaks.
    coordinates : array_like
        Positions of the identified kymograph peaks.
    model : callable
        Model which takes a particle's time and position and a list of prediction time points and then produces a mu
        and sigma for each of these future time points.
    sigma_cutoff: float
        At what fraction of sigma are points not going to be connected to this line at all?
    """
    score_matrix = -np.inf * np.ones((len(lines), len(coordinates)))

    for i, line in enumerate(lines):
        tip_time = line.time_idx[-1]
        tip_position = line.coordinate_idx[-1]

        mu_t, sigma_t = model(tip_time, tip_position, times)
        cutoff_lb = mu_t - sigma_cutoff * sigma_t
        cutoff_ub = mu_t + sigma_cutoff * sigma_t

        candidates = np.logical_and(coordinates > cutoff_lb, coordinates < cutoff_ub)

        score_matrix[i, candidates] = -(
            ((coordinates[candidates] - mu_t[candidates]) / sigma_t[candidates]) ** 2
        )

    return score_matrix


def kymo_score(vel=0, sigma=2, diffusion=0):
    """Return callable model used for computing linking scores.

    Model is comprised of a constant velocity (vel), an uncertainty (sigma) and a diffusion component (diffusion).
    Based on these three pieces of information, one can compute a mean and sigma for future time points given by:

        mu(t) = x + vel * t
        sigma(t) = sigma + sigma_diffusion * sqrt(t)

    These two values describe a probability density which is returned by the function.

    Parameters
    ----------
    vel: float
        mean velocity of the tracks.
    sigma: float
        noise around the track position.
    diffusion: float
        diffusion constant.
    """

    def prediction_model(t, coordinate, t_prediction):
        return kymo_diff_score(t, coordinate, t_prediction, vel, sigma, np.sqrt(2 * diffusion))

    return prediction_model
