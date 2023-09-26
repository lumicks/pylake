import warnings
from typing import List, Tuple, Optional
from dataclasses import field, dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DiffusionEstimate:
    """Diffusion estimate

    Attributes
    ----------
    value : float
        Estimate for the diffusion constant.
    std_err : float
        Standard error.

        Note: While this provides some measure of uncertainty, the estimate should not be used for
        calculating the weighted mean of multiple tracks. This is to prevent complications such as
        bias due to correlations between the estimated parameters and estimated variances. Instead,
        when calculating the weighted mean of estimates from time series of different lengths, the
        length :math:`N` of a time series should be used as weight, since it is known exactly.
    num_lags : Optional[int]
        Number of lags used to compute this estimate.
    num_points : int
        Number of points used to compute this estimate.
    localization_variance : float
        Estimated localization variance.

        Note: It is possible to obtain negative estimates for the localization variance when
        estimating diffusion based on single tracks.
    method : str
        String identifying which method was used to estimate the parameters.
    unit : str
        Unit that the diffusion constant is specified in.
    variance_of_localization_variance : Optional[float]
        Estimate of the variance of the estimated localization variance.
    """

    value: float
    std_err: float
    num_lags: Optional[int]
    num_points: int
    localization_variance: float
    method: str
    unit: str
    _unit_label: str = field(repr=False)
    variance_of_localization_variance: Optional[float] = None

    def __float__(self):
        return float(self.value)


@dataclass
class EnsembleMSD:
    r"""Ensemble MSD result

    Note that these values are obtained by using a weighted average of per-track MSDs. The
    weighting factor is determined by the number of points that went into the individual estimates.
    The standard error of the mean is computed using a weighted variance and the effective sample
    size determined for this procedure:

    .. math::

        SEM_{i} = \frac{\sigma_{i}}{\sqrt{N_{i, effective}}}

    with :math:`i` the lag index and :math:`N_{i, effective}` given by:

    .. math::

        N_{i, effective} = \frac{\left(\sum_{j}N_j\right)^2}{\sum_{j}N_{j}^2}

    with :math:`j` the track index. If all tracks are of equal size, the weighting will have no
    effect.

    Attributes
    ----------
    lags : np.ndarray
        Lags at which the MSD was computed.
    msd : np.ndarray
        Mean MSD for each lag.
    sem :  np.ndarray
        Standard error of the mean corresponding to each MSD.
    variance : np.ndarray
        Variance of each MSD average.
    counts : np.ndarray
        Number of elements that contributed to the estimate corresponding to each lag.
    effective_sample_size : np.ndarray
        Effective sample size.

        Since the estimate is based on weighted data, each observation does not contribute equally
        to the data. The effective sample size indicates the number of observations from an equally
        weighted sample that would yield the same level of precision. If all tracks have equal
        length and no missing data points, the effective sample size will simply equal the number of
        tracks.
    unit : str
        Unit that the diffusion constant is specified in.
    """

    lags: np.ndarray
    msd: np.ndarray
    sem: np.ndarray
    variance: np.ndarray
    counts: np.ndarray
    effective_sample_size: np.ndarray
    unit: str
    _time_step: float = field(repr=False)  # Time step in seconds
    _unit_label: str = field(repr=False)  # Unit in TeX format used for plotting labels.

    @property
    def seconds(self):
        return self.lags * self._time_step

    def plot(self, **kwargs):
        """Plot Ensemble MSDs

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.errorbar`.
        """
        import matplotlib.pyplot as plt

        plt.errorbar(self.seconds, self.msd, self.sem, **kwargs)
        plt.xlabel("Time [s]")
        plt.ylabel(f"Squared Displacement [{self._unit_label}]")


def weighted_mean_and_sd(means, counts) -> Tuple[float, float, float, float]:
    """Compute weighted mean, variance and effective sample size for a number of means.

    Computes the weighted mean and variance of a number of means with unequal samples contributing
    to them. Note that this function also returns the effective sample size which is required if
    you want to compute the standard error of the mean.

    Parameters
    ----------
    means : array_like
        List of means.
    counts : array_like
        Number of samples used to compute means.
    """
    if len(counts) <= 1:
        raise ValueError("Need more than one average to compute a weighted variance")

    if len(means) != len(counts):
        raise ValueError("Mean and count arrays must be the same size")

    counts_sum = np.sum(counts)
    weighted_mean = np.sum(means * counts) / counts_sum

    # The variance also involves weighting the individual means R by w = N / sum(N). When computing
    # the variance of this mean estimate, we need to do a bias correction because our effective
    # sample size is smaller than our total number of samples. This bias correction is given by:
    #
    #   1 / (1 - sum(w**2) / sum(w)**2)
    #
    # In other words, correcting for an effective sample size of sum(w)**2 / sum(w**2)
    #
    # We don't explicitly normalize the weights first. So what we end up with is the following
    # equation:
    #
    #   (sum(N) / (sum(N)**2 - sum(N**2)) * sum(N * (R - mean(R))**2)
    counts_squared_sum = np.sum(counts**2)
    normalization_constant = counts_sum / (counts_sum**2 - counts_squared_sum)
    weighted_variance = np.sum(counts * (means - weighted_mean) ** 2) * normalization_constant
    effective_sample_size = counts_sum**2 / counts_squared_sum

    return weighted_mean, weighted_variance, float(counts_sum), float(effective_sample_size)


def calculate_msd_counts(
    frame_idx, position, max_lag
) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """Compute mean squared displacements (MSDs) and counts (see calculate_msd).

    This function returns the mean squared displacement per lag along with the number of points
    that were used to estimate it.

    Parameters
    ----------
    frame_idx : array_like
        List of frame indices (note that these have to be of integral type to prevent rounding
        errors).
    position : array_like
        List of positions.
    max_lag : int
        Maximum number of lags to include (note that MSD estimates generally do not get better by
        including several lag steps).
    """
    frame_mesh_1, frame_mesh_2 = np.meshgrid(frame_idx, frame_idx)
    frame_diff = frame_mesh_1 - frame_mesh_2
    frame_lags = np.unique(frame_diff)

    position_mesh_1, position_mesh_2 = np.meshgrid(position, position)
    squared_displacements = (position_mesh_1 - position_mesh_2) ** 2

    # Look up only the rho elements we need
    frame_lags = frame_lags[frame_lags > 0][:max_lag]
    msds = [np.mean(squared_displacements[frame_diff == delta_frame]) for delta_frame in frame_lags]
    counts_per_lag = [np.sum(frame_diff == delta_frame) for delta_frame in frame_lags]

    return frame_lags, msds, counts_per_lag


def merge_track_msds(
    lags_msds_counts, min_count=0
) -> Tuple[npt.ArrayLike, List[Tuple[npt.ArrayLike, npt.ArrayLike]]]:
    r"""Aggregate unique lag, mean squared displacements and counts belonging to different tracks.

    This function takes a list of tuples with lags, MSDs and counts per track and combines these.
    The result is a list of unique lags and a list of MSD values and number of samples associated
    with each lag.

    Parameters
    ----------
    lags_msds_counts : list[tuple(array_like, array_like, array_like)]
        Individual lags, MSD estimates and number of samples used to calculate each MSD for
        multiple tracks.
    min_count : int
        Minimum count. Lags with fewer MSDs than `min_count` are omitted.
    """
    flattened_lags, flattened_msd_values, flattened_counts = (
        np.hstack([m[idx] for m in lags_msds_counts]) for idx in (0, 1, 2)
    )
    lags, inverse, counts_per_unique_lag = np.unique(
        flattened_lags, return_inverse=True, return_counts=True
    )

    filtered_lags = lags[counts_per_unique_lag >= min_count]

    # Collect the values for msd and count per lag
    msds_counts_per_lag = [
        (flattened_msd_values[inverse == idx], flattened_counts[inverse == idx])
        for idx, count in enumerate(counts_per_unique_lag)
        if count >= min_count
    ]

    return filtered_lags, msds_counts_per_lag


def calculate_msd(frame_idx, position, max_lag):
    r"""Estimate the Mean Squared Displacement (MSD) for various time lags.

    The estimator for the MSD (:math:`\rho`) is defined as:

    .. math::

      \rho_n = \frac{1}{N-n} \sum_{i=1}^{N-n}\left(r_{i+n} - r_{i}\right)^2

    here :math:`N` refers to the total frames, :math:`n` to the lag time and :math:`r_i` to the
    spatial position at lag :math:`i`. This function produces a list of lag times and mean squared
    displacements for those lag times.

    Parameters
    ----------
    frame_idx : array_like
        List of frame indices (note that these have to be of integral type to prevent rounding
        errors).
    position : array_like
        List of positions.
    max_lag : int
        Maximum number of lags to include (note that MSD estimates generally do not get better by
        including several lag steps).
    """
    frame_lags, msd_estimates, _ = calculate_msd_counts(frame_idx, position, max_lag)
    return frame_lags, msd_estimates


def calculate_ensemble_msd(
    line_msds, time_step, unit="au", unit_label="au", min_count=2
) -> EnsembleMSD:
    """Calculate ensemble MSDs.

    Parameters
    ----------
    line_msds : list of tuple of ndarray
        A list of tuples with three arrays. The three arrays are:
            - lags : lags at which the estimator is computed.
            - msds : MSD values.
            - counts : number of values that went into the estimate.
    time_step : float
        Time step in seconds.
    unit : str
        Spatial unit
    unit_label : str
        Spatial unit intended for the figure label
    min_count : int
        If fewer than `min_count` tracks contribute to the MSD at a particular lag then that lag
        is omitted
    """
    if len(line_msds) < 2:
        raise ValueError("You need at least two tracks to compute the ensemble MSD")

    lags, msds_counts_per_lag = merge_track_msds(line_msds, min_count)
    stats_per_lag = [weighted_mean_and_sd(msd, count) for msd, count in msds_counts_per_lag]
    mean, variance, counts, effective_sample_size = np.vstack(stats_per_lag).T

    return EnsembleMSD(
        lags=lags,
        msd=mean,
        sem=np.sqrt(variance / effective_sample_size),
        variance=variance,
        counts=counts,
        effective_sample_size=effective_sample_size,
        unit=f"{unit}^2",
        _time_step=time_step,
        _unit_label=f"{unit_label}²",
    )


def _msd_diffusion_covariance(max_lags, n, intercept, slope):
    """Covariance matrix for the mean squared displacements.

    Equation 8a from Bullerjahn et al [1]_.

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

    References
    ----------
    .. [1] Bullerjahn, J. T., von Bülow, S., & Hummer, G. (2020). Optimal estimates of
           self-diffusion coefficients from molecular dynamics simulations. The Journal of Chemical
           Physics, 153(2), 024116.
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


def _diffusion_ols(lag_idx, mean_squared_displacements, num_points):
    """Estimate the intercept, slope and standard deviation of the slope based on the msd [2]_

    Parameters
    ----------
    lag_idx : array_like
        lag time in frames
    mean_squared_displacements : array_like
        mean squared displacements to fit
    num_points : int
        number of points used to compute the lags

    References
    ----------
    .. [2] Bullerjahn, J. T., von Bülow, S., & Hummer, G. (2020). Optimal estimates of
           self-diffusion coefficients from molecular dynamics simulations. The Journal of Chemical
           Physics, 153(2), 024116.
    """
    num_lags = len(mean_squared_displacements)
    alpha = float(np.sum(lag_idx))
    beta = float(np.sum(lag_idx**2))

    # Estimate intercept and slope (Eq 5 from [2])
    gamma = np.sum(mean_squared_displacements)
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


def _update_gls_estimate(inverse_cov, mean_squared_displacements, intercept, slope):
    """Update the GLS estimates based on the inverted covariance matrix, current estimates and
    mean squared displacements.

    Parameters
    ----------
    inverse_cov : np.ndarray
        Inverted covariance matrix.
    mean_squared_displacements : array_like
        Mean squared displacements
    intercept : float
        Current estimate for the intercept
    slope : float
        Current estimate for the slope
    """
    num_lags = len(mean_squared_displacements)
    lag_idx = np.arange(num_lags) + 1
    j = np.tile(lag_idx, (num_lags, 1))
    i = j.T

    kappa = np.sum(inverse_cov)
    lam = np.sum(i * inverse_cov)
    mu = np.sum(i * j * inverse_cov)
    nu = np.sum(mean_squared_displacements * inverse_cov)
    xi = np.sum(i * mean_squared_displacements * inverse_cov)

    inv_denominator = 1.0 / (kappa * mu - lam**2)
    new_intercept = (mu * nu - lam * xi) * inv_denominator
    new_slope = (kappa * xi - lam * nu) * inv_denominator
    change = abs(new_intercept - intercept) + abs(new_slope - slope)

    var_slope = kappa / (kappa * mu - lam**2)

    return change, new_slope, new_intercept, var_slope


def _diffusion_gls(lag_idx, mean_squared_displacements, num_points, tolerance=1e-4, max_iter=100):
    """Estimate the intercept, slope and standard deviation of the slope based on the msd

    This method takes into account the covariance matrix and thereby does not suffer from including
    more lags than the optimal number of lags [3]_.

    Note that this method requires that no frames are missing from the result.

    Parameters
    ----------
    lag_idx : array_like
        lag time in frames
    mean_squared_displacements : array_like
        mean squared displacements to fit
    num_points : int
        number of points used to compute the lags
    tolerance : float
        termination tolerance for the iterative estimation
    max_iter : int
        maximum number of iterations

    References
    ----------
    .. [3] Bullerjahn, J. T., von Bülow, S., & Hummer, G. (2020). Optimal estimates of
           self-diffusion coefficients from molecular dynamics simulations. The Journal of Chemical
           Physics, 153(2), 024116.
    """
    # Eq. A1a from Appendix A of [3].
    num_lags = len(mean_squared_displacements)

    # Fetch initial guess for the iterative refinement (Appendix C from [3]).
    intercept = 2.0 * mean_squared_displacements[0] - mean_squared_displacements[1]
    slope = mean_squared_displacements[1] - mean_squared_displacements[0]

    def fallback(warning_message):
        """Fallback method if the GLS fails"""
        warnings.warn(RuntimeWarning(f"{warning_message} Reverting to two-point OLS."))
        return _diffusion_ols(lag_idx[:2], mean_squared_displacements[:2], num_points)

    # Since the covariance matrix depends on the parameters for the intercept and slope, we obtain
    # an implicit formulation. We use fixed point iteration to determine the parameters. If the
    # fixed point iteration fails to converge we fall back to the OLS solution for two points.
    for _ in range(max_iter):
        covariance_matrix = _msd_diffusion_covariance(num_lags, num_points, intercept, slope)

        # Solve generalized least squares problem using the current estimate for the covariance
        # matrix (Equation 10 from [3]).
        try:
            inverse_cov = np.linalg.inv(covariance_matrix)
        except np.linalg.LinAlgError:
            # Singular matrix, return OLS estimate
            return fallback("Covariance matrix is singular.")

        change, slope, intercept, var_slope = _update_gls_estimate(
            inverse_cov, mean_squared_displacements, intercept, slope
        )

        if change < tolerance:
            break
    else:
        return fallback("Maximum iterations reached!")

    return intercept, slope, var_slope


def estimate_diffusion_constant_simple(
    frame_idx,
    coordinate,
    time_step,
    max_lag,
    method,
    unit="au",
    unit_label="au",
):
    r"""Estimate diffusion constant

    The estimator for the MSD (:math:`\rho`) is defined as:

    .. math::

        \rho_n = \frac{1}{N-n} \sum_{i=1}^{N-n}\left(r_{i+n} - r_{i}\right)^2

    In a diffusion problem, the MSD can be fitted to a linear curve.

    .. math::

        intercept =& 2 d \left(\sigma^2 - 2 R D \Delta t\right)

        slope =& 2 d D \Delta t

    Here :math:`d` is the dimensionality of the problem. :math:`D` is the diffusion constant.
    :math:`R` is a motion blur constant. :math:`\Delta t` is the time step and sigma represents the
    dynamic localization error.

    One aspect that is import to consider is that this estimator uses every data point multiple
    times. As a consequence the elements of :math:`\rho_n` are highly correlated. This means that
    including more points doesn't necessarily make the estimates better and can actually make
    the estimate worse.

    There are two ways around this. Either you determine an optimal number of points to use
    in the estimation procedure (ols) [4]_ or you take into account the covariances present in
    the mean squared difference estimates (gls) [5]_.

    Note that this estimation procedure should only be used for pure diffusion in the absence
    of drift.

    Parameters
    ----------
    frame_idx : array_like
        Frame indices of the observations.
    coordinate : array_like
        Positional coordinates.
    time_step : float
        Time step between each frame [s].
    max_lag : int (optional)
        Number of lags to include. When omitted, the method will choose an appropriate number
        of lags to use.
        When the method chosen is "ols" an optimal number of lags is estimated as determined by
        [4]_. When the method is set to "gls" all lags are included.
    method : str
        Valid options are "ols" and "gls'.

        - "ols" : Ordinary least squares [4]_. Determines optimal number of lags.
        - "gls" : Generalized least squares [5]_. Takes into account covariance matrix (slower).
    unit : str
        Unit of the diffusion constant.
    unit_label : str
        Tex label for the unit of the diffusion constant.

    References
    ----------
    .. [4] Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
           single-particle tracking. Physical Review E, 85(6), 061916.
    .. [5] Bullerjahn, J. T., von Bülow, S., & Hummer, G. (2020). Optimal estimates of
           self-diffusion coefficients from molecular dynamics simulations. The Journal of Chemical
           Physics, 153(2), 024116.
    """
    if method not in ("gls", "ols"):
        raise ValueError('Invalid method selected. Method must be "gls" or "ols"')

    if not np.issubdtype(frame_idx.dtype, np.integer):
        raise TypeError("Frame indices need to be integer")

    if max_lag < 2:
        raise ValueError("You need at least two lags to estimate a diffusion constant")

    if any(np.diff(frame_idx) > 1):
        if method == "gls":
            raise RuntimeError(
                "Your tracks cannot have missing frames when using the GLS estimator. Refine your "
                "tracks using `lk.refine_tracks_centroid()` or `lk.refine_tracks_gaussian()`. "
                "Please refer to `help(lk.refine_tracks_centroid)` or "
                "`help(lk.refine_tracks_gaussian())` for more information."
            )
        elif method == "ols":
            warnings.warn(
                RuntimeWarning(
                    "Your tracks have missing frames. Note that this results in a poor estimate of "
                    "the standard error of the estimate. To avoid this warning, you can refine "
                    "your tracks using `lk.refine_tracks_centroid()` or "
                    "`lk.refine_tracks_gaussian()`. Please refer to "
                    "`help(lk.refine_tracks_centroid)` or `help(lk.refine_tracks_gaussian)` for "
                    "more information."
                ),
                stacklevel=2,
            )

    frame_lags, msd = calculate_msd(frame_idx, coordinate, max_lag)

    method_fun = _diffusion_gls if method == "gls" else _diffusion_ols
    intercept, slope, var_slope = method_fun(frame_lags, msd, len(coordinate))

    to_time = 1.0 / (2.0 * time_step)
    return DiffusionEstimate(
        value=slope * to_time,
        std_err=np.sqrt(var_slope) * to_time,
        num_lags=max_lag,
        num_points=len(coordinate),
        localization_variance=intercept / 2.0,
        method=method,
        unit=unit,
        _unit_label=unit_label,
    )


def calculate_localization_error(frame_lags, msd):
    r"""Determines the localization error, a metric used in the computation of the optimal number
    of points to include in the ordinary least squares estimate. The localization error is defined
    as:

    .. math::

        localization\_error = \frac{intercept}{slope} = \frac{sigma^2}{D \Delta t} - 2 R

    Parameters
    ----------
    frame_lags : array_like
        frame lags to include.
    msd : array_like
        (Correlated) Mean Squared Distance estimates as obtained by `calculate_msd`.

    Returns
    -------
    localization_error : float
        Localization error given by the ratio of the localization uncertainty and diffusive motion.

    Raises
    ------
    ValueError
        If the number of frame lags provided is not the same as the number of MSD values.
    """
    if len(frame_lags) != len(msd):
        raise ValueError(
            "To calculate a localization error, you need to supply an MSD estimate in msd for each "
            "lag time in frame_lag."
        )

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
    to estimate slope or intercept. These equations [6]_ minimize the relative error.

    References
    ----------
    .. [6] Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
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
    """Calculate optimal number of points to include in the diffusion estimate according to [7]_.

    Including more lags than necessary in an ordinary least squares estimate leads to excessive
    variance in the estimator due to the samples going into the MSD being highly correlated.
    Michalet et al. [7]_ provide a heuristic for determining the optimal number of points for
    different diffusion constants based on theoretical considerations. For more information, please
    refer to the paper.

    References
    ----------
    .. [7] Michalet, X., & Berglund, A. J. (2012). Optimal diffusion coefficient estimation in
           single-particle tracking. Physical Review E, 85(6), 061916.
    """
    if not np.issubdtype(frame_idx.dtype, np.integer):
        raise TypeError("Frame indices need to be integer")

    if any(np.diff(frame_idx) > 1):
        warnings.warn(
            RuntimeWarning(
                "Your tracks have missing frames. Note that this can lead to a suboptimal "
                "estimate of the optimal number of lags when using OLS."
            )
        )

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

        if len(coordinate) <= 4:
            raise RuntimeError(
                "You need at least 5 time points to estimate the number of points to include in "
                "the fit."
            )

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


def _var_cve_unknown_var(
    diffusion_constant, localization_var, dt, num_points, blur_constant=0, avg_frame_steps=1
) -> float:
    r"""Expected variance of the diffusion estimate obtained with CVE when the localization
    variance is not known a priori.

    The covariance-based diffusion estimator provides a simple unbiased estimator of diffusion.
    This function is based on equation 22 from Vestergaard [8]_ adapted for 1D. See also
    equation 17 from Vestergaard [9]_.

    Parameters
    ----------
    diffusion_constant : float
        Estimate of the diffusion constant
    localization_var : float
        Estimate of the localization variance
    dt : float
        Time step
    num_points : int
        Number of points in the trace
    blur_constant : float
        Motion blur constant. See :func:`lumicks.pylake.kymotracker.detail.msd_estimation._cve` for
        more information.
    avg_frame_steps : float
        Average frame steps. This number is the average of the time between two localizations in
        frames. This is given by:

        .. math::

            \frac{1}{N} \sum_{i=1}^{N_{frames - 1}} \frac{t_{i+1} - t_{i}}{dt}

        where :math:`dt` is the time step. If all frames had successful localization, this constant
        will be 1 [9]_.

    References
    ----------
    .. [8] Vestergaard, C. L. (2016). Optimizing experimental parameters for tracking of diffusing
           particles. Physical Review E, 94(2), 022401.
    .. [9] Vestergaard, C. L., Blainey, P. C., & Flyvbjerg, H. (2014). Optimal estimation of
           diffusion coefficients from single-particle trajectories. Physical Review E, 89(2),
           022726.
    """
    # Note that it uses a different definition of epsilon to circumvent a division by zero for D=0.
    epsilon = localization_var / dt - 2.0 * blur_constant * diffusion_constant
    avg_diff = avg_frame_steps * diffusion_constant
    numerator = 6.0 * avg_diff**2 + 4.0 * epsilon * avg_diff + 2.0 * epsilon**2
    denominator = num_points * avg_frame_steps**2
    term1 = numerator / denominator
    term2 = 4.0 * (avg_diff + epsilon) ** 2 / (num_points**2 * avg_frame_steps**2)
    return term1 + term2


def _var_cve_known_var(
    diffusion_constant,
    localization_var,
    var_of_localization_var,
    dt,
    num_points,
    blur_constant=0,
    avg_frame_steps=1,
) -> float:
    r"""Expected variance of the diffusion estimate obtained with CVE when the localization
    variance is known a-priori.

    The covariance-based diffusion estimator provides a simple unbiased estimator of diffusion.
    This function is based on equation 24 from Vestergaard [10]_ adapted for 1D. See also equation
    18 from Vestergaard [11]_.

    Parameters
    ----------
    diffusion_constant : float
        Estimate of the diffusion constant
    localization_var : float
        Estimate of the localization variance
    var_of_localization_var : float
        Variance of the localization variance estimate.
    dt : float
        Time step
    num_points : int
        Number of points in the trace
    blur_constant : float
        Motion blur constant. See :func:`lumicks.pylake.kymotracker.detail.msd_estimation._cve` for
        more information.
    avg_frame_steps : float
        Average frame steps. This number is the average of the time between two localizations in
        frames. This is given by:

        .. math::

            \frac{1}{N} \sum_{i=1}^{N_{frames - 1}} \frac{t_{i+1} - t_{i}}{dt}

        where :math:`dt` is the time step. If all frames had successful localization, this constant
        will be 1 [10]_.

    References
    ----------
    .. [10] Vestergaard, C. L. (2016). Optimizing experimental parameters for tracking of diffusing
            particles. Physical Review E, 94(2), 022401.
    .. [11] Vestergaard, C. L., Blainey, P. C., & Flyvbjerg, H. (2014). Optimal estimation of
            diffusion coefficients from single-particle trajectories. Physical Review E, 89(2),
            022726.
    """
    # Note that it uses a different definition of epsilon to circumvent a division by zero for D=0.
    epsilon = localization_var / dt - 2.0 * blur_constant * diffusion_constant
    blur_term = (avg_frame_steps - 2.0 * blur_constant) ** 2
    avg_diff = avg_frame_steps * diffusion_constant
    numerator = 2.0 * avg_diff**2 + 4.0 * epsilon * avg_diff + 3.0 * epsilon**2
    denominator = num_points * blur_term
    term1 = numerator / denominator
    term2 = var_of_localization_var / (blur_term * dt**2)
    return term1 + term2


def _cve(
    frame_indices, x, dt, blur_constant=0, localization_var=None, var_of_localization_var=None
) -> tuple:
    r"""Covariance based estimator.

    The covariance-based diffusion estimator provides a simple unbiased estimator of diffusion.
    This estimator was introduced in the work of Vestergaard [12]_. The correction for missing
    data was introduced in [13]_.

    Note that this estimator is only valid in the case of pure diffusion without drift.

    Parameters
    ----------
    frame_indices : array_like
        Frame indices
    x : array_like
        Positions over time
    dt : float
        Time step
    blur_constant : float
        Motion blur coefficient. Should be between 0 and 1/4 or `np.nan` if not clearly defined.

        The normalized shutter function is defined as :math:`c(t)`, where :math:`c(t)` represents
        whether the shutter is open or closed. :math:`c(t)` is normalized w.r.t. area. For no
        motion blur, :math:`c(t) = \delta(t_{\mathrm{exposure}})`, whereas for a constantly open
        shutter it is defined as :math:`c(t) = 1 / \Delta t`.

        With :math:`c(t)` defined as before, the motion blur constant is defined as:

        .. math::

            R = \frac{1}{\Delta t} \int_{0}^{\Delta t}S(t) \left(1 - S(t)\right)dt

        with

        .. math::

            S(t) = \int_{0}^{t} c(t') dt'

        When there is no motion blur, we obtain: R = 0, whereas a continuously open shutter over the
        exposure time results in R = 1/6. Note that when estimating both localization uncertainty
        and the diffusion constant, the motion blur factor has no effect on the estimate of the
        diffusion constant itself, but it does affect the calculated uncertainties. In the case of
        a provided localization uncertainty, it does impact the estimate of the diffusion constant.

        Note that if the blur constant is set to `np.nan`, not all estimates will be available.
    localization_var : Optional[float]
        Estimate of the localization error expressed as a variance, if it has been determined
        independently.
    var_of_localization_var : Optional[float]
        The variance of the localization variance estimate (needed if localization_var is provided).

    Note
    ----
    With the SNR defined as:

    .. math::

        \sqrt{D \Delta t} / \sigma

    With :math:`D` the diffusion constant and :math:`\sigma`
    This method requires SNR > 1. Below SNR = 1 it degrades rapidly [12]_.

    References
    ----------
    .. [12] Vestergaard, C. L., Blainey, P. C., & Flyvbjerg, H. (2014). Optimal estimation of
            diffusion coefficients from single-particle trajectories. Physical Review E, 89(2),
            022726.
    .. [13] Vestergaard, C. L. (2016). Optimizing experimental parameters for tracking of diffusing
            particles. Physical Review E, 94(2), 022401.
    """
    if not 0 <= blur_constant <= 0.25 and not np.isnan(blur_constant):
        raise ValueError("Motion blur constant should be between 0 and 1/4")

    if len(x) < 3:
        raise RuntimeError("Insufficient intervals for using the CVE")

    # Determine the average step size
    average_frame_step = np.mean(np.diff(frame_indices))
    avg_dt = average_frame_step * dt

    dx = np.diff(x)
    mean_dx_squared = np.mean(dx**2)

    if not localization_var:
        # Estimate the variance based on this track
        mean_dx_consecutive = np.mean(dx[1:] * dx[:-1])
        # Equation 14 from [12] and 21 from [13] adapted for 1D.
        diffusion_constant = mean_dx_squared / (2 * avg_dt) + mean_dx_consecutive / avg_dt
        # Equation 15 from [12]. Note that static loc uncertainty is not affected by frame skipping.
        localization_var = (
            blur_constant * mean_dx_squared + (2 * blur_constant - 1) * mean_dx_consecutive
        )
        diffusion_var = _var_cve_unknown_var(
            diffusion_constant, localization_var, dt, len(x), blur_constant, average_frame_step
        )
    else:
        if var_of_localization_var is None:
            raise ValueError(
                "When the localization variance is provided, the variance of this estimate should "
                "also be provided"
            )

        # We know the variance in advance. Equation 16 from [12] and 23 from [13] adapted for 1D.
        diffusion_constant = (mean_dx_squared - 2.0 * localization_var) / (
            2.0 * (avg_dt - 2.0 * blur_constant * dt)
        )
        diffusion_var = _var_cve_known_var(
            diffusion_constant,
            localization_var,
            var_of_localization_var,
            dt,
            len(x),
            blur_constant,
            average_frame_step,
        )

    return diffusion_constant, diffusion_var, localization_var


def estimate_diffusion_cve(
    frame_idx,
    coordinate,
    dt,
    blur_constant,
    unit,
    unit_label,
    localization_var=None,
    var_of_localization_var=None,
) -> DiffusionEstimate:
    """Estimate diffusion constant based on covariance estimator

    This function estimates the diffusion constant based on the covariance estimator and packs
    the result in a `DiffusionEstimate`. See the docstring for `_cve` for more information.

    Parameters
    ----------
    frame_idx : array_like
        Frame indices
    coordinate : array_like
        Positions over time
    dt : float
        Time step
    blur_constant : float
        Motion blur coefficient. Should be between 0 and 1/4 or `np.nan` if not defined. Note that
        if `np.nan` is specified, not all estimates will be available.
    unit : str
        Unit that the diffusion constant is specified in.
    unit_label : str
        Unit in TeX format used for plotting labels.
    localization_var : Optional[float]
        Estimated localization variance.
    var_of_localization_var : Optional[float]
        Estimated variance of the localization variance estimate.
    """

    diffusion, diffusion_var, localization_var = _cve(
        frame_idx, coordinate, dt, blur_constant, localization_var, var_of_localization_var
    )
    return DiffusionEstimate(
        value=diffusion,
        std_err=np.sqrt(diffusion_var),
        num_lags=None,
        num_points=len(coordinate),
        localization_variance=localization_var,
        method="cve",
        unit=unit,
        _unit_label=unit_label,
        variance_of_localization_variance=var_of_localization_var,
    )


def ensemble_cve(kymotracks, calculate_localization_var=True):
    """Calculate ensemble-based CVE.

    Determines the weighted average of the mean and standard error of the CVE estimates.

    See docstring for _cve for more information on the estimator itself. The averaging is handled
    through equation 57 and 58 from [14]_.

    Parameters
    ----------
    kymotracks : lumicks.pylake.kymotracker.kymotrack.KymoTrackGroup
        Group of kymotracks
    calculate_localization_var : bool
        If localization variance can be calculated. Must be False if source kymographs
        do not have the same line times or pixel sizes.

    Warns
    -----
    RuntimeWarning
        if `method == "cve"` and the source kymographs do not have the same line times
        or pixel sizes. As a result, the localization variance and variance of the localization
        variance not be available. Estimates that are unavailable are returned as `np.nan`.

    References
    ----------
    .. [14] Vestergaard, C. L., Blainey, P. C., & Flyvbjerg, H. (2014). Optimal estimation of
            diffusion coefficients from single-particle trajectories. Physical Review E, 89(2),
            022726.
    """
    cve_based = kymotracks.estimate_diffusion(method="cve")

    def mean_var(x, count, count_sum):
        # Equation 58 and 57 from Vestergaard et al [14].
        weighted_mean = np.sum(x * count) / count_sum
        weighted_mean_var = np.sum(count * (x - weighted_mean) ** 2) / (
            (count.size - 1) * count_sum
        )
        return weighted_mean, weighted_mean_var

    if len(cve_based) == 1:
        return cve_based[0]

    counts = np.array([c.num_points for c in cve_based])
    counts_sum = sum(counts)

    ensemble_mean, ensemble_mean_var = mean_var(
        np.array([c.value for c in cve_based]), counts, counts_sum
    )

    ensemble_localization_var, var_of_localization_var = (
        mean_var(np.array([c.localization_variance for c in cve_based]), counts, counts_sum)
        if calculate_localization_var
        else (np.nan, np.nan)
    )

    return DiffusionEstimate(
        value=ensemble_mean,
        std_err=np.sqrt(ensemble_mean_var),
        num_lags=np.nan,
        num_points=counts_sum,
        localization_variance=ensemble_localization_var,
        method="ensemble cve",
        unit=cve_based[0].unit,
        _unit_label=cve_based[0].unit,
        variance_of_localization_variance=var_of_localization_var,
    )


def _determine_optimal_points_ensemble(frame_lags, msds, n_coord, max_iterations=100):
    """Calculate optimal number of points to include in the diffusion estimate.

    See :func:`lumicks.pylake.kymotracker.detail.msd_estimation.determine_optimal_points`
    for more information.
    """
    num_slope = max(2, n_coord // 10)  # Need at least two points for a linear regression!

    num_slopes = set()
    for _ in np.arange(max_iterations):
        num_slopes.add(num_slope)

        # Determine the number of points to include in the next fit
        num_slope, _ = optimal_points(
            calculate_localization_error(frame_lags[:num_slope], msds[:num_slope]), n_coord
        )

        if num_slope in num_slopes:
            return num_slope

    warnings.warn(
        RuntimeWarning("Warning, maximum number of iterations exceeded. Returning best solution.")
    )

    return num_slope


def ensemble_ols(kymotracks, max_lag):
    ensemble_msd = kymotracks.ensemble_msd(max_lag)
    track_length = len(ensemble_msd.counts) + 1

    if any(np.diff(ensemble_msd.lags) > 1):
        warnings.warn(
            RuntimeWarning(
                "Your tracks have missing frames. Note that this can lead to a suboptimal estimates"
            )
        )

    optimal_lags = (
        _determine_optimal_points_ensemble(ensemble_msd.lags, ensemble_msd.msd, track_length)
        if not max_lag
        else max_lag
    )

    intercept, slope, var_slope = _diffusion_ols(
        ensemble_msd.lags[:optimal_lags], ensemble_msd.msd[:optimal_lags], track_length
    )

    time_step = kymotracks._kymos[0].line_time_seconds
    to_time = 1.0 / (2.0 * time_step)

    src_calibration = kymotracks._kymos[0]._calibration
    return DiffusionEstimate(
        value=slope * to_time,
        std_err=np.sqrt(var_slope / np.mean(ensemble_msd.effective_sample_size)) * to_time,
        num_lags=optimal_lags,
        num_points=sum(len(t) for t in kymotracks),
        localization_variance=intercept / 2.0,
        method="ensemble ols",
        unit=f"{src_calibration.unit}^2 / s",
        _unit_label=f"{src_calibration.unit_label}²/s",
    )
