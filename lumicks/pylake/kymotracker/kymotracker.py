from .detail.trace_line_2d import detect_lines, points_to_line_segments
from .detail.scoring_functions import kymo_score
from .kymotrack import KymoTrack, KymoTrackGroup
from .detail.gaussian_mle import gaussian_mle_1d, overlapping_pixels
from .detail.peakfinding import (
    peak_estimate,
    refine_peak_based_on_moment,
    merge_close_peaks,
    KymoPeaks,
)
from .detail.localization_models import GaussianLocalizationModel
import numpy as np
import warnings
from deprecated.sphinx import deprecated

__all__ = [
    "track_greedy",
    "track_lines",
    "filter_tracks",
    "refine_tracks_centroid",
    "refine_tracks_gaussian",
    "filter_lines",
    "refine_lines_centroid",
    "refine_lines_gaussian",
]


_default_track_widths = {"um": 0.35, "kbp": 0.35 / 0.34, "pixel": 4}


def _to_pixel_rect(rect, pixelsize, line_time_seconds):
    """Convert (time, position) coordinates from physical units to pixels.

    Note: return values are rounded (toward zero) to integer pixel.

    Parameters
    ----------
    rect: array-like
        Array of (time, position) tuples in physical units; length 2.
    pixelsize: float
        Size of spatial pixel dimension in physical units.
    line_time_seconds: float
        Size of temporal pixel dimension in seconds.
    """
    return [
        [int(seconds / line_time_seconds), int(position / pixelsize)]
        for (seconds, position) in rect
    ]


def track_greedy(
    kymograph,
    channel,
    track_width=None,
    pixel_threshold=None,
    window=8,
    sigma=None,
    vel=0.0,
    diffusion=0.0,
    sigma_cutoff=2.0,
    rect=None,
    line_width=None,
):
    """Track particles on an image using a greedy algorithm.

    Note: This is ALPHA functionality. It has not been tested in a sufficient number of cases yet,
    and the API is still subject to change without a prior deprecation notice.

    A method based on connecting feature points. Detection of the feature points is done analogously
    to [1]_, using a greyscale dilation approach to detect peaks, followed by a local centroid
    computation to achieve subpixel accuracy. After peak detection the feature points are linked
    together using a greedy forward search analogous to [2]_. This in contrast with the linking
    algorithm in [1]_ which uses a graph based optimization approach.

    The linking step traverses the kymograph, tracing tracks starting from each frame. It starts with
    the highest intensity track and proceeds to tracks with lower signal intensity. For every point along the
    track, the algorithm makes a prediction for where the particle will be in the next frame. Points
    are considered candidates for track membership when they fall within a cone parameterized by a
    sigma and diffusion constant. The candidate point closest to the prediction is chosen and
    connected to the track. When no more candidates are available the track is terminated.

    *Note: the `track_width` parameter is given in physical units, but the algorithm works with discrete
    pixels. In order to avoid bias in the result, the number of pixels to use is rounded up to the
    nearest odd value.*

    Parameters
    ----------
    kymograph : lumicks.pylake.Kymo
        The kymograph to track.
    channel : {'red', 'green', 'blue'}
        Color channel to track.
    track_width : float or None
        Expected (spatial) spot size in physical units. Must be larger than zero.
        If `None`, the default is 0.35 (half the wavelength of the red limit of the visible spectrum)
        for kymographs calibrated in microns. For kymographs calibrated in kilobase pairs the
        corresponding value is calculated using 0.34 nm/bp (from duplex DNA).
    pixel_threshold : float or None
        Intensity threshold for the pixels. Local maxima above this intensity level will be
        designated as a track origin. Must be larger than zero. If `None`, the default is set to the
        98th percentile of the image signal.
    window : int
        Number of kymograph lines in which the particle is allowed to disappear (and still be part
        of the same track).
    sigma : float or None
        Uncertainty in the particle position. This parameter will determine whether a peak in the
        next frame will be linked to this one. Increasing this value will make the algorithm tend
        to allow more positional variation in the tracks. If none, the algorithm will use half the
        track width.
    vel : float
        Expected velocity of the traces in the image in physical units. This can be used for
        non-static particles that are expected to move at a constant rate (default: 0.0).
    diffusion : float
        Expected diffusion constant in physical units (default: 0.0). This parameter will influence
        whether a peak in the next frame will be connected to this one. Increasing this value will
        make the algorithm allow more positional variation over time. If a particle disappears for
        a few frames and then reappears, points before and after the interval where the particle
        was not visible are more likely to be connected with a higher diffusion setting. Must be
        equal to or greater than zero.
    sigma_cutoff : float
        Sets at how many standard deviations from the expected trajectory a particle no longer
        belongs to this track. Lower values result in tracks being more stringent in terms of
        continuing (default: 2.0).
    rect : tuple of two coordinates
        Only perform tracking over a subset of the image. When this argument is supplied, the peak
        detection and refinement is performed over the full image, but the results are then filtered
        to omit the peaks that fall outside of the rect. Coordinates should be given as:
        ((min_time, min_coord), (max_time, max_coord)).
    line_width : float
        **Deprecated** Forwarded to `track_width`.

    References
    ----------
    .. [1] Sbalzarini, I. F., & Koumoutsakos, P. (2005). Feature point tracking and trajectory
           analysis for video imaging in cell biology. Journal of structural biology, 151(2),
           182-195.
    .. [2] Mangeol, P., Prevo, B., & Peterman, E. J. (2016). KymographClear and KymographDirect: two
           tools for the automated quantitative analysis of molecular and cellular dynamics using
           kymographs. Molecular biology of the cell, 27(12), 1948-1957.
    """

    # TODO: remove line_width argument deprecation path
    if track_width is None:
        if line_width is None:
            track_width = _default_track_widths[kymograph._calibration.unit]
        else:
            track_width = line_width
            warnings.warn(
                DeprecationWarning(
                    "The argument `line_width` is deprecated; use `track_width` instead."
                ),
                stacklevel=2,
            )

    if pixel_threshold is None:
        pixel_threshold = np.percentile(kymograph.get_image(channel), 98)

    if track_width <= 0:
        # Must be positive otherwise refinement fails
        raise ValueError(f"track_width should be larger than zero")

    if pixel_threshold <= 0:
        raise ValueError(f"pixel_threshold should be larger than zero")

    if diffusion < 0:
        raise ValueError(f"diffusion should be positive")  # Must be positive or score model fails

    kymograph_data = kymograph.get_image(channel)

    position_scale = kymograph.pixelsize[0]
    track_width_pixels = track_width / position_scale

    coordinates, time_points = peak_estimate(
        kymograph_data, np.ceil(0.5 * track_width_pixels), pixel_threshold
    )
    if len(coordinates) == 0:
        return []

    position, time, m0 = refine_peak_based_on_moment(
        kymograph_data, coordinates, time_points, np.ceil(0.5 * track_width_pixels)
    )

    if rect:
        (t0, p0), (t1, p1) = _to_pixel_rect(
            rect, kymograph.pixelsize[0], kymograph.line_time_seconds
        )
        mask = (position >= p0) & (position < p1) & (time >= t0) & (time < t1)
        position, time, m0 = position[mask], time[mask], m0[mask]

    peaks = KymoPeaks(position, time, m0)
    peaks = merge_close_peaks(peaks, np.ceil(0.5 * track_width_pixels))

    # Convert algorithm parameters to pixel units
    velocity_pixels = vel * kymograph.line_time_seconds / position_scale
    diffusion_pixels = diffusion / (position_scale**2 / kymograph.line_time_seconds)
    sigma_pixels = sigma / position_scale if sigma else 0.5 * track_width_pixels

    tracks = points_to_line_segments(
        peaks,
        kymo_score(
            vel=velocity_pixels,
            sigma=sigma_pixels,
            diffusion=diffusion_pixels,
        ),
        window=window,
        sigma_cutoff=sigma_cutoff,
    )

    tracks = [
        KymoTrack(track.time_idx, track.coordinate_idx, kymograph, channel) for track in tracks
    ]

    return KymoTrackGroup(tracks)


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

    This method is based on sections 1, 2 and 3 from [1]_. This method attempts to find lines purely
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
        Expected line width in physical units. Must be larger than zero.
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
    .. [1] Steger, C. (1998). An unbiased detector of curvilinear structures. IEEE Transactions on
           pattern analysis and machine intelligence, 20(2), 113-125.
    """
    if line_width <= 0:
        raise ValueError("line_width should be larger than zero")

    kymograph_data = kymograph.get_image(channel)
    roi = (
        _to_pixel_rect(rect, kymograph.pixelsize[0], kymograph.line_time_seconds) if rect else None
    )

    lines = detect_lines(
        kymograph_data,
        line_width / kymograph.pixelsize[0],
        max_lines=max_lines,
        start_threshold=start_threshold,
        continuation_threshold=continuation_threshold,
        angle_weight=angle_weight,
        force_dir=1,
        roi=roi,
    )

    return KymoTrackGroup(
        [KymoTrack(line.time_idx, line.coordinate_idx, kymograph, channel) for line in lines]
    )


@deprecated(
    reason=("`filter_lines()` has been renamed to `filter_tracks()`."),
    action="always",
    version="0.13.0",
)
def filter_lines(lines, minimum_length):
    """Remove lines below a specific minimum number of points from the list.

    This can be used to enforce a minimum number of frames a spot has to be detected in to be
    considered a valid trace.
    """
    return filter_tracks(lines, minimum_length)


def filter_tracks(tracks, minimum_length):
    """Remove tracks shorter than a minimum number of time points from the list.

    This can be used to enforce a minimum number of frames a spot has to be detected in order
    to be considered a valid track.

    Parameters
    ----------
    tracks : List[pylake.KymoTrack]
        Detected tracks on a kymograph.
    minimum_length : int
        Minimum length for the track to be accepted.
    """
    return KymoTrackGroup([track for track in tracks if len(track) >= minimum_length])


@deprecated(
    reason=("`refine_lines_centroid()` has been renamed to `refine_tracks_centroid()`."),
    action="always",
    version="0.13.0",
)
def refine_lines_centroid(lines, line_width):
    """Refine the lines based on the brightness-weighted centroid.

    This function interpolates the determined traces and then uses the pixels in the vicinity of the
    traces to make small adjustments to the estimated location. The refinement correction is
    computed by considering the brightness weighted centroid.
    """
    if line_width < 1:
        # Refinement only does something when line_width in pixels is larger than 1
        raise ValueError("line_width may not be smaller than 1")

    # convert line_width (pixel units) to physical units expected by refine_tracks_centroid
    track_width = line_width * lines[0]._kymo.pixelsize[0]
    return refine_tracks_centroid(lines, track_width)


def refine_tracks_centroid(tracks, track_width=None):
    """Refine the tracks based on the brightness-weighted centroid.

    This function interpolates the determined tracks (in time) and then uses the pixels in the vicinity of the
    tracks to make small adjustments to the estimated location. The refinement correction is
    computed by considering the brightness weighted centroid.

    *Note: the `track_width` parameter is given in physical units, but the algorithm works with discrete
    pixels. In order to avoid bias in the result, the number of pixels to use is rounded up to the
    nearest odd value.*

    Parameters
    ----------
    tracks : List[pylake.KymoTrack]
        Detected tracks on a kymograph
    track_width : float
        Expected (spatial) spot size in physical units. Must be larger than zero.
        If `None`, the default is 0.35 (half the wavelength of the red limit of the visible spectrum)
        for kymographs calibrated in microns. For kymographs calibrated in kilobase pairs the
        corresponding value is calculated using 0.34 nm/bp (from duplex DNA).
    """
    if track_width is None:
        track_width = _default_track_widths[tracks[0]._kymo._calibration.unit]

    if track_width <= 0:
        # Must be positive otherwise refinement fails
        raise ValueError(f"track_width should be larger than zero")

    track_width_pixels = np.ceil(track_width / tracks[0]._kymo.pixelsize[0])

    interpolated_tracks = [track.interpolate() for track in tracks]
    time_idx = np.round(
        np.array(np.hstack([track.time_idx for track in interpolated_tracks]))
    ).astype(int)
    coordinate_idx = np.round(
        np.array(np.hstack([track.coordinate_idx for track in interpolated_tracks]))
    ).astype(int)

    coordinate_idx, time_idx, _ = refine_peak_based_on_moment(
        interpolated_tracks[0]._image.data,
        coordinate_idx,
        time_idx,
        np.ceil(0.5 * track_width_pixels),
    )

    track_ids = np.hstack(
        [np.full(len(track.time_idx), j) for j, track in enumerate(interpolated_tracks)]
    )
    new_tracks = [
        track._with_coordinates(time_idx[track_ids == j], coordinate_idx[track_ids == j])
        for j, track in enumerate(interpolated_tracks)
    ]
    return KymoTrackGroup(new_tracks)


@deprecated(
    reason=("`refine_lines_gaussian()` has been renamed to `refine_tracks_gaussian()`."),
    action="always",
    version="0.13.0",
)
def refine_lines_gaussian(
    lines,
    window,
    refine_missing_frames,
    overlap_strategy,
    initial_sigma=None,
    fixed_background=None,
):
    """Refine the lines by gaussian peak MLE."""
    return refine_tracks_gaussian(
        lines,
        window,
        refine_missing_frames,
        overlap_strategy,
        initial_sigma=initial_sigma,
        fixed_background=fixed_background,
    )


def refine_tracks_gaussian(
    tracks,
    window,
    refine_missing_frames,
    overlap_strategy,
    initial_sigma=None,
    fixed_background=None,
):
    """Refine the tracks by gaussian peak MLE.

    Parameters
    ----------
    tracks : List[pylake.KymoTrack] or pylake.KymoTrackGroup
        Detected tracks on a kymograph.
    window : int
        Number of pixels on either side of the estimated track to include in the optimization data.
        The fitting window should be large enough to capture the tails of the gaussian PSF, but
        ideally small enough such that it will not include data from nearby tracks.
    refine_missing_frames : bool
        Whether to estimate location for frames which were missed in initial peak finding.
    overlap_strategy : {'multiple', 'ignore', 'skip'}
        How to deal with frames in which the fitting window of two `KymoTrack`'s overlap.

        - 'multiple' : fit the peaks simultaneously.
        - 'ignore' : do nothing, fit the frame as-is (ignoring overlaps).
        - 'skip' : skip optimization of the frame; remove from returned `KymoTrack`.
    initial_sigma : float
        Initial guess for the `sigma` parameter.
    fixed_background : float
        Fixed background parameter in photons per second.
        When supplied, the background is not estimated but fixed at this value.
    """
    assert overlap_strategy in ("ignore", "skip", "multiple")
    if refine_missing_frames:
        tracks = [track.interpolate() for track in tracks]

    kymo = tracks[0]._kymo
    channel = tracks[0]._channel
    image_data = kymo.get_image(channel)

    initial_sigma = kymo.pixelsize[0] * 1.1 if initial_sigma is None else initial_sigma

    full_position = np.arange(image_data.shape[0]) * kymo.pixelsize[0]
    overlap_count = 0

    # Generate a structure in which we can look up which lines contribute to which frame
    # 3 groups: (spatial) pixel coordinate, spatial position, line index
    tracks_per_frame = [[[] for _ in range(image_data.shape[1])] for _ in range(3)]
    for track_index, track in enumerate(tracks):
        for idx, frame_index in enumerate(track.time_idx):
            tracks_per_frame[0][int(frame_index)].append(int(track.coordinate_idx[idx]))
            tracks_per_frame[1][int(frame_index)].append(track.position[idx])
            tracks_per_frame[2][int(frame_index)].append(track_index)
    tracks_per_frame = zip(*tracks_per_frame)

    # Prepare storage for the refined lines
    refined_tracks_time_idx = [[] for _ in range(len(tracks))]
    refined_tracks_parameters = [[] for _ in range(len(tracks))]

    for frame_index, (pixel_coordinates, positions, track_indices) in enumerate(tracks_per_frame):
        # Determine which lines are close enough so that they have to be fitted in the same group
        groups = (
            [[idx] for idx in range(len(pixel_coordinates))]
            if overlap_strategy == "ignore"
            else overlapping_pixels(pixel_coordinates, window)
        )
        for group in groups:
            if len(group) > 1 and overlap_strategy == "skip":
                overlap_count += 1
                continue

            # Grab the line indices within the group
            track_indices_group = [track_indices[idx] for idx in group]
            initial_positions = [positions[idx] for idx in group]

            # Cut out the relevant chunk of data
            limits = slice(
                max(pixel_coordinates[group[0]] - window, 0),
                pixel_coordinates[group[-1]] + window + 1,
            )
            photon_counts = image_data[limits, frame_index]

            result = gaussian_mle_1d(
                full_position[limits],
                photon_counts,
                kymo.pixelsize[0],
                initial_position=initial_positions,
                initial_sigma=initial_sigma,
                fixed_background=fixed_background,
            )

            # Store results in refined lines
            for track_idx, params in zip(track_indices_group, result):
                refined_tracks_time_idx[track_idx].append(frame_index)
                is_overlapping = len(result) != 1 if overlap_strategy == "multiple" else False
                refined_tracks_parameters[track_idx].append(np.hstack((params, is_overlapping)))

    if overlap_count and overlap_strategy != "ignore":
        warnings.warn(
            f"There were {overlap_count} instances of overlapped tracks ignored while fitting."
        )

    return KymoTrackGroup(
        [
            KymoTrack(t, GaussianLocalizationModel(*np.vstack(p).T), kymo, channel)
            for t, p in zip(refined_tracks_time_idx, refined_tracks_parameters)
            if len(t) > 0
        ]
    )
