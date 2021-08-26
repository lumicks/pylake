import numpy as np
import warnings


def calculate_msd(frame_idx, position, max_lag):
    """Estimate the Mean Square Displacement (MSD) for various time lags.

    The estimator for the MSD (rho) is defined as:

      rho_n = (1 / (N-n)) sum_{i=1}^{N-n}(r_{i+n} - r_{i})^2

    here N refers to the total frames, n to the lag time and r_i the spatial position at lag i.
    This function produces a list of lag times and mean squared displacements for those lag times.

    Parameters
    ----------
    frame_idx : array_like
        List of frame indices (note that these have to be of integral type to prevent rounding
        errors).
    position : array_like
        List of positions.
    max_lag : float
        Maximum lag to include (note that MSD estimates generally do not get better by including
        several lag steps).
    """
    frame_mesh_1, frame_mesh_2 = np.meshgrid(frame_idx, frame_idx)
    frame_diff = frame_mesh_1 - frame_mesh_2
    frame_lags = np.unique(frame_diff)

    position_mesh_1, position_mesh_2 = np.meshgrid(position, position)
    summand = (position_mesh_1 - position_mesh_2) ** 2

    # Look up only the rho elements we need
    frame_lags = frame_lags[frame_lags > 0][:max_lag]
    msd = np.array([np.mean(summand[frame_diff == delta_frame]) for delta_frame in frame_lags])

    return frame_lags, msd


def estimate_diffusion_constant_simple(frame_idx, coordinate, time_step, max_lag):
    """Perform an unweighted fit to the MSD estimates to obtain a diffusion constant.

    The estimator for the MSD (rho) is defined as:

      rho_n = (1 / (N-n)) sum_{i=1}^{N-n}(r_{i+n} - r_{i})^2

    In a diffusion problem, the MSD can be fitted to a linear curve.

        intercept = 2 * d * (sigma**2 - 2 * R * D * delta_t)
        slope = 2 * d * D * delta_t

    Here d is the dimensionality of the problem (in this case, d is set to 1). D is the diffusion
    constant. R is a motion blur constant. delta_t is the time step and sigma represents the dynamic
    localization error (which is not necessarily the same as the static localization error).

    One aspect that is import to consider is that this estimator uses every data point multiple
    times. As a consequence the elements of rho_n are highly correlated. This means that including
    more points doesn't necessarily make the estimates better and can actually make the estimate
    worse. It is therefore a good idea to estimate an appropriate number of MSD estimates to use.
    See [1] for more information on this.

    Note that this estimation procedure should only be used for Brownian motion in isotropic
    media (meaning no cellular or structured environments) in the absence of drift.

    1) Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
    single-particle tracking. Physical Review E, 85(6), 061916.

    Parameters
    ----------
    frame_idx : array_like
        Frame indices of the observations.
    coordinate : array_like
        Positional coordinates.
    time_step : float
        Time step between each frame.
    max_lag : int
        Maximum delay to include in the estimate (must be larger than 1).
    """
    if not np.issubdtype(frame_idx.dtype, np.integer):
        raise TypeError("Frame indices need to be integer")

    if max_lag < 2:
        raise ValueError("You need at least two lags to estimate a diffusion constant")

    frame_lags, msd = calculate_msd(frame_idx, coordinate, max_lag)
    coefficients = np.polyfit(frame_lags, msd, 1)
    return coefficients[0] / (2.0 * time_step)


def calculate_localization_error(frame_lags, msd):
    """Determines the localization error, a metric used in the computation of the optimal number
    of points to include in the ordinary least squares estimate. The localization error is defined
    as:

        localization_error = intercept / slope = sigma**2 / (D * delta_t) - 2 * R

    Parameters
    ----------
    frame_lags : array_like
        frame lags to include.
    msd : array_like
        (Correlated) Mean Squared Distance estimates as obtained by `calculate_msd`.
    """
    assert len(frame_lags) == len(msd), "Need to supply an MSD estimate per lag time"
    coefficients = np.polyfit(frame_lags, msd, 1)
    slope, intercept = coefficients[0], coefficients[1]

    if intercept < 0:
        return 0
    elif slope < 0:
        return np.inf
    else:
        return intercept / slope


def optimal_points(localization_error, num_points):
    """Empirical relationship described in Michalet et al for determining optimal number of points
    to estimate slope or intercept. These equations minimize the relative error. See [1] for more
    information.

    1) Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
    single-particle tracking. Physical Review E, 85(6), 061916.
    """
    if num_points <= 4:
        raise RuntimeError(
            "You need at least 5 time points to estimate the number of points to include in the "
            "fit."
        )

    fa = 2.0 + 1.6 * localization_error ** 0.51
    limit_a = 3 + (4.5 * num_points ** 0.4 - 8.5) ** 1.2

    fb = 2 + 1.35 * localization_error ** 0.6
    limit_b = 0.8 + 0.564 * num_points

    if np.isinf(localization_error):
        num_points_intercept, num_points_slope = np.floor(limit_a), np.floor(limit_b)
    else:
        num_points_intercept = np.floor(fa * limit_a / (fa ** 3 + limit_a ** 3) ** (1.0 / 3.0))
        num_points_slope = min(
            np.floor(limit_b), np.floor(fb * limit_b / (fb ** 3 + limit_b ** 3) ** (1.0 / 3.0))
        )

    return max(2, int(num_points_slope)), max(2, int(num_points_intercept))


def determine_optimal_points(frame_idx, coordinate, max_iterations=100):
    """Calculate optimal number of points to include in the diffusion estimate according to [1].

    Including more lags than necessary in an ordinary least squares estimate leads to excessive
    variance in the estimator due to the samples going into the MSD being highly correlated. The
    equations in [1] provide a heuristic for determining the optimal number of points for different
    diffusion constants based on theoretical considerations. For more information, please refer
    to the paper.

    1) Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
    single-particle tracking. Physical Review E, 85(6), 061916.
    """
    if not np.issubdtype(frame_idx.dtype, np.integer):
        raise TypeError("Frame indices need to be integer")

    num_slope = max(2, len(coordinate) // 10)  # Need at least two points for a linear regression!
    num_intercept = max(2, len(coordinate) // 10)
    number_computed = 0

    num_slopes = set()
    for _ in np.arange(max_iterations):
        # Only evaluate what we need
        required_points = max(num_intercept, num_slope)
        if required_points > number_computed:
            frame_lags, msd = calculate_msd(frame_idx, coordinate, required_points)
            number_computed = required_points

        num_slopes.add(num_slope)

        # Determine the number of points to include in the next fit
        num_slope, num_intercept = optimal_points(
            calculate_localization_error(frame_lags[:num_slope], msd[:num_slope]), len(coordinate)
        )

        if num_slope in num_slopes:
            return num_slope, num_intercept

    warnings.warn(
        RuntimeWarning("Warning, maximum number of iterations exceeded. Returning best solution.")
    )
    return num_slope, num_intercept
