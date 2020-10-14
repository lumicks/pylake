from lumicks.pylake.kymotracker.detail.peakfinding import peak_estimate, refine_peak_based_on_moment, merge_close_peaks
from lumicks.pylake.kymotracker.detail.trace_line_2d import detect_lines, points_to_line_segments
from lumicks.pylake.kymotracker.detail.scoring_functions import kymo_score
import numpy as np


def _get_rect(data, rect):
    ((t0, p0), (t1, p1)) = rect

    if p0 > data.shape[0]:
        raise IndexError(f"Specified minimum position {p0} beyond the valid coordinate range {data.shape[0]}")

    if t0 > data.shape[1]:
        raise IndexError(f"Specified minimum time {t0} beyond the time range {data.shape[1]}")

    if t0 >= t1:
        raise IndexError("Please specify rectangle from minimum time to maximum time")

    if p0 >= p1:
        raise IndexError("Please specify rectangle from minimum position to maximum position")

    return data[p0:p1, t0:t1]


def track_greedy(data, line_width, pixel_threshold, window=8, sigma=None, vel=0.0, diffusion=0.0, sigma_cutoff=2.0,
                 rect=None):
    """Track particles on an image using a greedy algorithm.

    Note: This is ALPHA functionality. It has not been tested in a sufficient number of cases yet, and the API is still
    subject to change without a prior deprecation notice.

    A method based on connecting feature points. Detection of the feature points is done analogously to [1], using
    a greyscale dilation approach to detect peaks, followed by a local centroid computation to achieve subpixel
    accuracy. After peak detection the feature points are linked together using a greedy forward search analogous to
    [2]. This in contrast with the linking algorithm in [1] which uses a graph based optimization approach.

    The linking step traverses the kymograph, tracing lines starting from each frame. It starts with the highest line
    and proceeds to lines with lower signal intensity. For every point along the line, the algorithm makes a prediction
    for where the particle will be in the next frame. Points are considered candidates for line membership when they
    fall within a cone parameterized by a sigma and diffusion constant. The candidate point closest to the prediction
    is chosen and connected to the line. When no more candidates are available the line is terminated.

    Parameters
    ----------
    data : array_like
        N by M image containing a single color channel.
    line_width : float
        Expected line width in pixels.
    pixel_threshold : float
        Intensity threshold for the pixels. Local maxima above this intensity level will be designated as a line
        origin.
    window : int
        Number of kymograph lines in which the particle is allowed to disappear (and still be part of the same line).
    sigma : float or None
        Uncertainty in the particle position. This parameter will determine whether a peak in the next frame will be
        linked to this one. Increasing this value will make the algorithm tend to allow more positional variation in
        the lines. If none, the algorithm will use half the line width.
    vel : float
        Expected velocity of the traces in the image. This can be used for non-static particles that are expected to
        move at an expected rate (default: 0.0).
    diffusion : float
        Expected diffusion constant (default: 0.0). This parameter will influence whether a peak in the next frame
        will be connected to this one. Increasing this value will make the algorithm allow more positional variation
        in.
    sigma_cutoff : float
        Sets at how many standard deviations from the expected trajectory a particle no longer belongs to this trace.
        Lower values result in traces being more stringent in terms of continuing (default: 2.0).
    rect : tuple of two pixel coordinates
        Only perform tracking over a subset of the image. Pixel coordinates should be given as:
        ((min_time, min_coord), (max_time, max_coord)).

    References
    ----------
    [1] Sbalzarini, I. F., & Koumoutsakos, P. (2005). Feature point tracking and trajectory analysis for video
    imaging in cell biology. Journal of structural biology, 151(2), 182-195.
    [2] Mangeol, P., Prevo, B., & Peterman, E. J. (2016). KymographClear and KymographDirect: two tools for the
    automated quantitative analysis of molecular and cellular dynamics using kymographs. Molecular biology of the
    cell, 27(12), 1948-1957.
    """
    data_selection = _get_rect(data, rect) if rect else data

    sigma = sigma if sigma else .5 * line_width
    coordinates, time_points = peak_estimate(data_selection, np.ceil(.5*line_width), pixel_threshold)
    if len(coordinates) == 0:
        return []

    peaks = refine_peak_based_on_moment(data_selection, coordinates, time_points, np.ceil(.5*line_width))
    peaks = merge_close_peaks(peaks, np.ceil(.5*line_width))
    lines = points_to_line_segments(
        peaks,
        kymo_score(vel=vel, sigma=sigma, diffusion=diffusion),
        window=window,
        sigma_cutoff=sigma_cutoff,
    )

    # Note that this deliberately refers to the original data, not the tracked subset!
    for line in lines:
        line.image_data = data

    return [line.with_offset(rect[0][0], rect[0][1]) for line in lines] if rect else lines


def track_lines(data, line_width, max_lines, start_threshold=0.005, continuation_threshold=0.005, angle_weight=10.0,
                rect=None):
    """Track particles on an image using an algorithm that looks for line-like structures.

    Note: This is ALPHA functionality. It has not been tested in a sufficient number of cases yet, and the API is still
    subject to change without a prior deprecation notice.

    This function tracks particles in an image. It takes a pixel image, and traces lines on it. These lines can
    subsequently be refined and/or used to extract intensities or other parameters.

    This method is based on sections 1, 2 and 3 from [1]. This method attempts to find lines purely based on
    differential geometric considerations. It blurs the image based with a user specified line width and then attempts
    to find curvilinear sections. Based on eigenvalue decomposition of the local Hessian it finds the principal
    direction of the line. It then computes subpixel accurate positions by computing the maximum perpendicular
    to the line using local Taylor expansions. Pixels where this subpixel position estimate falls within the pixel are
    considered lines. The initial step leads to lines with a width of one. These lines are then traced. When ambiguity
    arises, on which point to connect next, a score comprised of the distance to the next subpixel minimum and angle
    between the successive normal vectors is computed. The candidate with the lowest score is then selected.

    For more information, please refer to the paper.

    Parameters
    ----------
    data : array_like
        N by M image containing a single color channel.
    line_width : float
        Expected line width in pixels.
    max_lines : int
        Maximum number of lines to trace.
    start_threshold : float
        Threshold for the value of the derivative.
    continuation_threshold : float
        Derivative threshold for the continuation of a line.
    angle_weight: float
        Factor which determines how the angle between normals needs to be weighted relative to distance.
        High values push for straighter lines. Weighting occurs according to distance + angle_weight * angle difference
    rect : tuple of two pixel coordinates
        Only perform tracking over a subset of the image. Pixel coordinates should be given as:
        ((min_time, min_coord), (max_time, max_coord)).

    References
    ----------
    [1] Steger, C. (1998). An unbiased detector of curvilinear structures. IEEE Transactions on pattern analysis and
    machine intelligence, 20(2), 113-125.
    """

    lines = detect_lines(
        _get_rect(data, rect) if rect else data,
        line_width,
        max_lines=max_lines,
        start_threshold=start_threshold,
        continuation_threshold=continuation_threshold,
        angle_weight=angle_weight,
        force_dir=1,
    )

    # Note that this deliberately refers to the original data, not the tracked subset!
    for line in lines:
        line.image_data = data

    return [line.with_offset(rect[0][0], rect[0][1]) for line in lines] if rect else lines


def filter_lines(lines, minimum_length):
    """Remove lines below a specific minimum number of points from the list

    Parameters
    ----------
    lines : List[pylake.KymoLine]
        Detected traces on a kymograph.
    minimum_length : int
        Minimum length for the line to be accepted.
    """
    return [line for line in lines if len(line) >= minimum_length]
