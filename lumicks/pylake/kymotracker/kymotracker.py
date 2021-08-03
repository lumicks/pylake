from .detail.calibrated_images import CalibratedKymographChannel
from .detail.trace_line_2d import detect_lines, points_to_line_segments
from .detail.scoring_functions import kymo_score
from .kymoline import KymoLine
from .detail.gaussian_mle import gaussian_mle_1d
from .detail.peakfinding import (
    peak_estimate,
    refine_peak_based_on_moment,
    merge_close_peaks,
    KymoPeaks,
)
from .kymoline import KymoLineGroup
import numpy as np
import warnings

__all__ = [
    "track_greedy",
    "track_lines",
    "filter_lines",
    "refine_lines_centroid",
    "refine_lines_gaussian",
]


def track_greedy(
    kymograph,
    channel,
    line_width,
    pixel_threshold,
    window=8,
    sigma=None,
    vel=0.0,
    diffusion=0.0,
    sigma_cutoff=2.0,
    rect=None,
):
    """Track particles on an image using a greedy algorithm.

    Note: This is ALPHA functionality. It has not been tested in a sufficient number of cases yet,
    and the API is still subject to change without a prior deprecation notice.

    A method based on connecting feature points. Detection of the feature points is done analogously
    to [1], using a greyscale dilation approach to detect peaks, followed by a local centroid
    computation to achieve subpixel accuracy. After peak detection the feature points are linked
    together using a greedy forward search analogous to [2]. This in contrast with the linking
    algorithm in [1] which uses a graph based optimization approach.

    The linking step traverses the kymograph, tracing lines starting from each frame. It starts with
    the highest line and proceeds to lines with lower signal intensity. For every point along the
    line, the algorithm makes a prediction for where the particle will be in the next frame. Points
    are considered candidates for line membership when they fall within a cone parameterized by a
    sigma and diffusion constant. The candidate point closest to the prediction is chosen and
    connected to the line. When no more candidates are available the line is terminated.

    Parameters
    ----------
    kymograph : lumicks.pylake.Kymo
        Kymograph.
    channel : str
        Kymograph channel.
    line_width : float
        Expected line width in physical units.
    pixel_threshold : float
        Intensity threshold for the pixels. Local maxima above this intensity level will be
        designated as a line origin.
    window : int
        Number of kymograph lines in which the particle is allowed to disappear (and still be part
        of the same line).
    sigma : float or None
        Uncertainty in the particle position. This parameter will determine whether a peak in the
        next frame will be linked to this one. Increasing this value will make the algorithm tend
        to allow more positional variation in the lines. If none, the algorithm will use half the
        line width.
    vel : float
        Expected velocity of the traces in the image in physical units. This can be used for
        non-static particles that are expected to move at an expected rate (default: 0.0).
    diffusion : float
        Expected diffusion constant in physical units (default: 0.0). This parameter will influence
        whether a peak in the next frame will be connected to this one. Increasing this value will
        make the algorithm allow more positional variation over time. If a particle disappears for
        a few frames and then reappears, points before and after the interval where the particle
        was not visible are more likely to be connected with a higher diffusion setting.
    sigma_cutoff : float
        Sets at how many standard deviations from the expected trajectory a particle no longer
        belongs to this trace. Lower values result in traces being more stringent in terms of
        continuing (default: 2.0).
    rect : tuple of two coordinates
        Only perform tracking over a subset of the image. When this argument is supplied, the peak
        detection and refinement is performed over the full image, but the results are then filtered
        to omit the peaks that fall outside of the rect. Coordinates should be given as:
        ((min_time, min_coord), (max_time, max_coord)).

    References
    ----------
    [1] Sbalzarini, I. F., & Koumoutsakos, P. (2005). Feature point tracking and trajectory analysis
    for video imaging in cell biology. Journal of structural biology, 151(2), 182-195.
    [2] Mangeol, P., Prevo, B., & Peterman, E. J. (2016). KymographClear and KymographDirect: two
    tools for the automated quantitative analysis of molecular and cellular dynamics using
    kymographs. Molecular biology of the cell, 27(12), 1948-1957.
    """
    kymograph_channel = CalibratedKymographChannel.from_kymo(kymograph, channel)

    position_scale = kymograph_channel._pixel_size
    line_width_pixels = line_width / position_scale

    coordinates, time_points = peak_estimate(
        kymograph_channel.data, np.ceil(0.5 * line_width_pixels), pixel_threshold
    )
    if len(coordinates) == 0:
        return []

    position, time, m0 = refine_peak_based_on_moment(
        kymograph_channel.data, coordinates, time_points, np.ceil(0.5 * line_width_pixels)
    )

    if rect:
        (t0, p0), (t1, p1) = kymograph_channel._to_pixel_rect(rect)
        mask = (position >= p0) & (position < p1) & (time >= t0) & (time < t1)
        position, time, m0 = position[mask], time[mask], m0[mask]

    peaks = KymoPeaks(position, time, m0)
    peaks = merge_close_peaks(peaks, np.ceil(0.5 * line_width_pixels))

    # Convert algorithm parameters to pixel units
    velocity_pixels = vel * kymograph_channel.line_time_seconds / position_scale
    diffusion_pixels = diffusion / (position_scale ** 2 / kymograph_channel.line_time_seconds)
    sigma_pixels = sigma / position_scale if sigma else 0.5 * line_width_pixels

    lines = points_to_line_segments(
        peaks,
        kymo_score(
            vel=velocity_pixels,
            sigma=sigma_pixels,
            diffusion=diffusion_pixels,
        ),
        window=window,
        sigma_cutoff=sigma_cutoff,
    )

    lines = [KymoLine(line.time_idx, line.coordinate_idx, kymograph_channel) for line in lines]

    return KymoLineGroup(lines)


def track_lines(
    kymograph,
    channel,
    line_width,
    max_lines,
    start_threshold=0.005,
    continuation_threshold=0.005,
    angle_weight=10.0,
    rect=None,
):
    """Track particles on an image using an algorithm that looks for line-like structures.

    Note: This is ALPHA functionality. It has not been tested in a sufficient number of cases yet,
    and the API is still subject to change without a prior deprecation notice.

    This function tracks particles in an image. It takes a pixel image, and traces lines on it.
    These lines can subsequently be refined and/or used to extract intensities or other parameters.

    This method is based on sections 1, 2 and 3 from [1]. This method attempts to find lines purely
    based on differential geometric considerations. It blurs the image based with a user specified
    line width and then attempts to find curvilinear sections. Based on eigenvalue decomposition of
    the local Hessian it finds the principal direction of the line. It then computes subpixel
    accurate positions by computing the maximum perpendicular to the line using local Taylor
    expansions. Pixels where this subpixel position estimate falls within the pixel are considered
    lines. The initial step leads to lines with a width of one. These lines are then traced. When
    ambiguity arises, on which point to connect next, a score comprised of the distance to the next
    subpixel minimum and angle between the successive normal vectors is computed. The candidate
    with the lowest score is then selected.

    For more information, please refer to the paper.

    Parameters
    ----------
    kymograph : lumicks.pylake.Kymo
        Kymograph.
    channel : str
        Kymograph channel.
    line_width : float
        Expected line width in physical units.
    max_lines : int
        Maximum number of lines to trace.
    start_threshold : float
        Threshold for the value of the derivative.
    continuation_threshold : float
        Derivative threshold for the continuation of a line.
    angle_weight: float
        Factor which determines how the angle between normals needs to be weighted relative to
        distance. High values push for straighter lines. Weighting occurs according to
        distance + angle_weight * angle difference
    rect : tuple of two coordinates
        Only perform tracking over a subset of the image. Coordinates should be given as:
        ((min_time, min_coord), (max_time, max_coord)).

    References
    ----------
    [1] Steger, C. (1998). An unbiased detector of curvilinear structures. IEEE Transactions on
    pattern analysis and machine intelligence, 20(2), 113-125.
    """
    kymograph_channel = CalibratedKymographChannel.from_kymo(kymograph, channel)

    lines = detect_lines(
        kymograph_channel.data,
        line_width / kymograph_channel._pixel_size,
        max_lines=max_lines,
        start_threshold=start_threshold,
        continuation_threshold=continuation_threshold,
        angle_weight=angle_weight,
        force_dir=1,
        roi=kymograph_channel._to_pixel_rect(rect) if rect else None,
    )

    return KymoLineGroup(
        [KymoLine(line.time_idx, line.coordinate_idx, kymograph_channel) for line in lines]
    )


def filter_lines(lines, minimum_length):
    """Remove lines below a specific minimum number of points from the list.

    This can be used to enforce a minimum number of frames a spot has to be detected in to be
    considered a valid trace.

    Parameters
    ----------
    lines : List[pylake.KymoLine]
        Detected traces on a kymograph.
    minimum_length : int
        Minimum length for the line to be accepted.
    """
    return KymoLineGroup([line for line in lines if len(line) >= minimum_length])


def refine_lines_centroid(lines, line_width):
    """Refine the lines based on the brightness-weighted centroid.

    This function interpolates the determined traces and then uses the pixels in the vicinity of the
    traces to make small adjustments to the estimated location. The refinement correction is
    computed by considering the brightness weighted centroid.

    Parameters
    ----------
    lines : List[pylake.KymoLine]
        Detected traces on a kymograph.
    line_width : int
        Line width
    max_iter : int
        Maximum number of iterations
    """
    interpolated_lines = [line.interpolate() for line in lines]
    time_idx = np.round(np.array(np.hstack([line.time_idx for line in interpolated_lines]))).astype(
        int
    )
    coordinate_idx = np.round(
        np.array(np.hstack([line.coordinate_idx for line in interpolated_lines]))
    ).astype(int)

    coordinate_idx, time_idx, _ = refine_peak_based_on_moment(
        interpolated_lines[0]._image.data, coordinate_idx, time_idx, np.ceil(0.5 * line_width)
    )

    current = 0
    for line in interpolated_lines:
        line_length = len(line.time_idx)
        line.time_idx = time_idx[current : current + line_length]
        line.coordinate_idx = coordinate_idx[current : current + line_length]
        current += line_length

    return KymoLineGroup(interpolated_lines)


def refine_lines_gaussian(
    lines, window, refine_missing_frames, overlap_strategy, initial_sigma=None
):
    """Refine the lines by gaussian peak MLE.

    Parameters
    ----------
    lines : List[pylake.KymoLine] or pylake.KymolineGroup
        Detected traces on a kymograph.
    window : int
        Number of pixels on either side of the estimated line to include in the optimization data.
    refine_missing_frames : bool
        Whether to estimate location for frames which were missed in initial peak finding.
    overlap_strategy : {'ignore', 'skip'}
        How to deal with frames in which the fitting window of two `KymoLine`'s overlap.

        - 'ignore' : do nothing, fit the frame as-is.
        - 'skip' : skip optimization of the frame; remove from returned `KymoLine`.
    initial_sigma : float
        Initial guess for the `sigma` parameter.
    """
    assert overlap_strategy in ("ignore", "skip")
    if refine_missing_frames:
        lines = [line.interpolate() for line in lines]

    image = lines[0]._image
    initial_sigma = image._pixel_size * 1.1 if initial_sigma is None else initial_sigma
    get_window_limits = lambda center: (max(int(center) - window, 0), int(center) + window + 1)
    get_pixel_indices = lambda line: zip(line.time_idx.astype(int), np.rint(line.coordinate_idx))

    # locate pixels with overlapping track windows
    image_mask = np.zeros(image.data.shape)
    for line in lines:
        for j, (frame_index, coordinate_index) in enumerate(get_pixel_indices(line)):
            limits = slice(*get_window_limits(coordinate_index))
            image_mask[limits, frame_index] += 1

    refined_lines = KymoLineGroup([])
    full_position = image.to_position(np.arange(image.data.shape[0]))
    overlap_count = 0

    for line in lines:
        tmp_times = []
        tmp_positions = []
        for j, (frame_index, coordinate_index) in enumerate(get_pixel_indices(line)):
            limits = slice(*get_window_limits(coordinate_index))

            overlap = image_mask[limits, frame_index] >= 2
            if np.any(overlap) and overlap_strategy == "skip":
                overlap_count += 1
                continue
            position = full_position[limits]
            photon_counts = image.data[limits, frame_index]

            r = gaussian_mle_1d(
                position,
                photon_counts,
                image._pixel_size,
                initial_position=line.position[j],
                initial_sigma=initial_sigma,
            )
            tmp_positions.append(r.x[1] / image._pixel_size)
            tmp_times.append(line.time_idx[j])

        if len(tmp_positions):
            refined_lines.extend(KymoLine(tmp_times, tmp_positions, image))

    if overlap_count and overlap_strategy != "ignore":
        warnings.warn(
            f"There were {overlap_count} instances of overlapped tracks ignored while fitting."
        )
    return refined_lines
