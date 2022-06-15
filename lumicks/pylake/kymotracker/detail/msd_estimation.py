import numpy as np
import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionEstimate:
    """Diffusion estimate

    Attributes
    ----------
    value : float
        Estimate for the diffusion constant.
    std_err : float
        Standard error,
    num_lags : int
        Number of lags used to compute this estimate.
    num_points : int
        Number of points used to compute this estimate.
    method : str
        String identifying which method was used to estimate the parameters.
    """

    value: float
    std_err: float
    num_lags: int
    num_points: int
    method: str

    def __float__(self):
        return float(self.value)


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


def _msd_diffusion_covariance(max_lags, n, intercept, slope):
    """Covariance matrix for the mean squared displacements.

    Equation 8a from [2].

    Parameters
    ----------
    max_lags : max_lags
        number of lags used in estimation.
    n : int
        number of trajectory points used
    intercept : float
        estimated localization uncertainty
    slope : float
        estimate for the slope

    2) Bullerjahn, J. T., von Bülow, S., & Hummer, G. (2020). Optimal estimates of self-diffusion
    coefficients from molecular dynamics simulations. The Journal of Chemical Physics, 153(2),
    024116.
    """
    # Intercept corresponds to a^2 in the paper, slope refers to sigma^2 in the paper
    i = np.tile(np.arange(max_lags) + 1, (max_lags, 1))
    j = i.T
    min_ij = np.minimum(i, j)

    # Covariance matrix for localization uncertainty = 0
    term1 = 2.0 * min_ij * (1.0 + 3.0 * i * j - min_ij**2) / (n - min_ij + 1)
    denominator = (n - i + 1.0) * (n - j + 1.0)
    term2 = (min_ij**2 - min_ij**4) / denominator
    heaviside = (i + j - n - 2) >= 0
    term3 = heaviside * ((n + 1.0 - i - j) ** 4 - (n + 1.0 - i - j) ** 2) / denominator

    # Covariance matrix if there was no localization uncertainty (Eq 8b)
    base_covariance = (slope**2 / 3) * (term1 + term2 + term3)

    # Intercept corresponds to a^2 in the paper, slope refers to sigma^2 in the paper
    term4_numerator = intercept**2 * (1.0 + (i == j)) + 4 * intercept * slope * min_ij
    term4_denominator = n - min_ij + 1.0
    term5_numerator = intercept**2 * np.maximum(0.0, n - i - j + 1.0)
    term5_denominator = (n - i + 1.0) * (n - j + 1.0)
    localization_part = term4_numerator / term4_denominator + term5_numerator / term5_denominator

    # Equation 8a
    return base_covariance + localization_part


def _diffusion_ols(mean_squared_displacements, num_points):
    """Estimate the intercept, slope and standard deviation of the slope based on the msd

    Parameters
    ----------
    mean_squared_displacements : array_like
        mean squared displacements to fit
    num_points : int
        number of points used to compute the lags

    2) Bullerjahn, J. T., von Bülow, S., & Hummer, G. (2020). Optimal estimates of self-diffusion
    coefficients from molecular dynamics simulations. The Journal of Chemical Physics, 153(2),
    024116.
    """
    num_lags = len(mean_squared_displacements)
    alpha = num_lags * (num_lags + 1.0) * 0.5
    beta = alpha * (2.0 * num_lags + 1.0) / 3.0

    # Estimate intercept and slope (Eq 5 from [2])
    gamma = np.sum(mean_squared_displacements)
    lag_idx = np.arange(num_lags) + 1
    delta = np.sum(lag_idx * mean_squared_displacements)
    inv_denominator = 1.0 / (num_lags * beta - alpha**2)
    intercept = (beta * gamma - alpha * delta) * inv_denominator
    slope = (num_lags * delta - alpha * gamma) * inv_denominator

    # Determine variance on slope estimator (Eq. A1a from Appendix A of [2]).
    covariance_matrix = _msd_diffusion_covariance(num_lags, num_points, intercept, slope)

    i = np.tile(lag_idx, (num_lags, 1))
    j = i.T
    numerator = (i * num_lags - alpha) * (j * num_lags - alpha) * covariance_matrix
    denominator = (num_lags * beta - alpha**2) ** 2
    var_slope = np.sum(numerator / denominator)

    return intercept, slope, var_slope


def estimate_diffusion_constant_simple(frame_idx, coordinate, time_step, max_lag, method):
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

    The standard deviation of the diffusion estimate is obtained using the equations for the OLS
    estimator from [2].

    Note that this estimation procedure should only be used for Brownian motion in isotropic
    media (meaning no cellular or structured environments) in the absence of drift.

    1) Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
    single-particle tracking. Physical Review E, 85(6), 061916.
    2) Bullerjahn, J. T., von Bülow, S., & Hummer, G. (2020). Optimal estimates of self-diffusion
    coefficients from molecular dynamics simulations. The Journal of Chemical Physics, 153(2),
    024116.

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
    method : str
        Should be "ols".
    """
    if not np.issubdtype(frame_idx.dtype, np.integer):
        raise TypeError("Frame indices need to be integer")

    if max_lag < 2:
        raise ValueError("You need at least two lags to estimate a diffusion constant")

    frame_lags, msd = calculate_msd(frame_idx, coordinate, max_lag)
    _, slope, var_slope = _diffusion_ols(msd, len(coordinate))
    to_time = 1.0 / (2.0 * time_step)
    return DiffusionEstimate(
        slope * to_time, np.sqrt(var_slope) * to_time, max_lag, len(coordinate), method
    )


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

    fa = 2.0 + 1.6 * localization_error**0.51
    limit_a = 3 + (4.5 * num_points**0.4 - 8.5) ** 1.2

    fb = 2 + 1.35 * localization_error**0.6
    limit_b = 0.8 + 0.564 * num_points

    if np.isinf(localization_error):
        num_points_intercept, num_points_slope = np.floor(limit_a), np.floor(limit_b)
    else:
        num_points_intercept = np.floor(fa * limit_a / (fa**3 + limit_a**3) ** (1.0 / 3.0))
        num_points_slope = min(
            np.floor(limit_b), np.floor(fb * limit_b / (fb**3 + limit_b**3) ** (1.0 / 3.0))
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
