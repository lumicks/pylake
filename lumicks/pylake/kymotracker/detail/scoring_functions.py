import numpy as np


def cone_score(x, t, vel, sigma, sigma_diffusion):
    """This function returns an estimates mean and sigma for future time points based on a velocity, fixed uncertainty
    and diffusion rate. This is used to compute a scoring function to determine whether two local maxima should be
    linked.

    We expect the future to be a combination of a base uncertainty, a diffusion and a constant velocity. The stdev for
    the particle at time point t for the diffusion process is given by:
        sigma(t) = N(mu(t), sigma*sqrt(t)).
    mu is simply given by the prediction based on velocity.

    Parameters
    ----------
        x : array_like
            Coordinates
        t : array_like
            Time points
        vel : float
            Estimated velocity of the particle
        sigma : float
            Base positional variability
        sigma_diffusion : float
            Sigma representing diffusion. For diffusion, the spread is characterized by:
                Sigma(t) = sigma_diffusion * sqrt(t)
            Note that sigma_diffusion equates to sqrt(2*D) where D is the diffusion constant.
    """

    assert np.all(t > 0)
    mu_t = x + vel * t
    sigma_t = sigma + sigma_diffusion * np.sqrt(t)

    return mu_t, sigma_t


def build_score_matrix(lines, times, coordinates, sigma, sigma_diffusion, sigma_cutoff, vel):
    """Builds a score matrix for a combination of lines and positions. The score matrix contains a score for each
    line, point pair. For each line, we calculate a penalty function which reflects a score associated with connecting
    those two lines. In the current implementation these are based on a Gaussian cone around the most likely trajectory.
    Sigma provides a starting width, whereas diffusion widens the cone over time.

    Since the score is simply used as a relative scoring metric, and we are not interested in the absolute values,
    we do not have to exponentiate the Gaussian.

    Above a certain cutoff value (sigma_cutoff), connections are considered very unlikely and candidates in those
    regions are considered to not be able to be part of the same line.

    Parameters
    ----------
    lines: list of pylake.Line
    times: array_like
        Time points corresponding to the identified kymograph peaks.
    coordinates : array_like
        Positions of the identified kymograph peaks.
    sigma: float
        Starting uncertainty of the cone.
    sigma_diffusion:
        Sigma representing diffusion. For diffusion, the spread is characterized by:
          Sigma(t) = sigma_diffusion * sqrt(t)
        Note that sigma_diffusion equates to sqrt(2*D) where D is the diffusion constant.
    sigma_cutoff: float
        At what fraction of sigma are points not going to be connected to this line at all?
    vel: float
        mean velocity of moving particles.
    """
    score_matrix = -np.inf * np.ones((len(lines), len(coordinates)))

    for i, line in enumerate(lines):
        tip_time = line.time[-1]
        tip_position = line.coordinate[-1]
        temporal_diff = times - tip_time

        # Calculate probability cone
        mu_t, sigma_t = cone_score(tip_position, temporal_diff, vel, sigma, sigma_diffusion)
        cutoff_lb = mu_t - sigma_cutoff * sigma_t
        cutoff_ub = mu_t + sigma_cutoff * sigma_t

        candidates = np.logical_and(coordinates > cutoff_lb, coordinates < cutoff_ub)

        score_matrix[i, candidates] = -((coordinates[candidates] - mu_t[candidates]) / sigma_t[candidates]) ** 2

    return score_matrix
