import warnings
from itertools import chain

import numpy as np

from .kymotrack import KymoTrack, KymoTrackGroup
from .detail.peakfinding import find_kymograph_peaks, refine_peak_based_on_moment
from .detail.gaussian_mle import gaussian_mle_1d, overlapping_pixels
from .detail.trace_line_2d import detect_lines, points_to_line_segments
from .detail.scoring_functions import kymo_score
from .detail.localization_models import GaussianLocalizationModel

__all__ = [
    "track_greedy",
    "track_lines",
    "filter_tracks",
    "refine_tracks_centroid",
    "refine_tracks_gaussian",
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


def _to_half_kernel_size(width, pixelsize):
    """Converts a window width to a half kernel size.

    This function is used to calculate how many pixels to add on each side of a center pixel to
    generate an odd kernel used for filtering."""
    return np.ceil(width / pixelsize).astype(int) // 2


def _validate_track_width(track_width, bound, unit):
    # We lower the minimum bound slightly to ensure that setting it to the minimum bound exactly
    # becomes a valid value.
    if track_width < np.nextafter(bound, 0):
        raise ValueError(f"track_width must at least be 3 pixels ({bound:.3f} [{unit}])")


def track_greedy(
    kymograph,
    channel,
    *,
    track_width=None,
    filter_width=None,
    pixel_threshold=None,
    window=8,
    adjacency_filter=False,
    sigma=None,
    velocity=0.0,
    diffusion=0.0,
    sigma_cutoff=2.0,
    rect=None,
    bias_correction=True,
):
    """Track particles on an image using a greedy algorithm.

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

    .. note::

        The `track_width` parameter is given in physical units, but the algorithm works with
        discrete pixels. In order to avoid bias in the result, the number of pixels to use is
        rounded up to the nearest odd value.

    Parameters
    ----------
    kymograph : Kymo
        The kymograph to track.
    channel : {'red', 'green', 'blue'}
        Color channel to track.
    track_width : float or None
        Expected (spatial) spot size in physical units. Must be larger than zero.
        If `None`, the default is 0.35 (half the wavelength of the red limit of the visible spectrum)
        for kymographs calibrated in microns. For kymographs calibrated in kilobase pairs the
        corresponding value is calculated using 0.34 nm/bp (from duplex DNA).

        Note: For tracking purposes, the track width is rounded up to the nearest odd number of
        pixels; the resulting value must be at least 3 pixels.
    filter_width : float or None
        Filter width in microns. Should ideally be chosen to the width of the PSF (default: None which
        results in half a pixel size for legacy reasons).
    pixel_threshold : float or None
        Intensity threshold for the pixels. Local maxima above this intensity level will be
        designated as a track origin. Must be larger than zero. If `None`, the default is set to the
        98th percentile of the image signal.
    window : int
        Number of kymograph lines in which the particle is allowed to disappear (and still be part
        of the same track).
    adjacency_filter : bool
        Require that any true peak detection has a positive detection in an adjacent frame.
    sigma : float or None
        Uncertainty in the particle position. This parameter will determine whether a peak in the
        next frame will be linked to this one. Increasing this value will make the algorithm tend
        to allow more positional variation in the tracks. If none, the algorithm will use half the
        track width.
    velocity : float
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
    bias_correction : bool
        Correct coordinate bias by ensuring that the window is symmetric (see [3]_ for more
        information). Note that while this increases the variance (reduces precision), it can
        greatly reduce the bias when there is a high background.

    References
    ----------
    .. [1] Sbalzarini, I. F., & Koumoutsakos, P. (2005). Feature point tracking and trajectory
           analysis for video imaging in cell biology. Journal of structural biology, 151(2),
           182-195.
    .. [2] Mangeol, P., Prevo, B., & Peterman, E. J. (2016). KymographClear and KymographDirect: two
           tools for the automated quantitative analysis of molecular and cellular dynamics using
           kymographs. Molecular biology of the cell, 27(12), 1948-1957.
    .. [3] Berglund, A. J., McMahon, M. D., McClelland, J. J., & Liddle, J. A. (2008).
           Fast, bias-free algorithm for tracking single particles with variable size and
           shape. Optics express, 16(18), 14064-14075.
    """
    if track_width is None:
        track_width = max(
            _default_track_widths[kymograph._calibration.unit], 3 * kymograph.pixelsize[0]
        )

    if pixel_threshold is None:
        pixel_threshold = np.percentile(kymograph.get_image(channel), 98)

    _validate_track_width(track_width, 3 * kymograph.pixelsize[0], kymograph._calibration.unit)

    if pixel_threshold <= 0:
        raise ValueError("pixel_threshold should be larger than zero")

    if diffusion < 0:
        raise ValueError("diffusion should be positive")  # Must be positive or score model fails

    kymograph_data = kymograph.get_image(channel)

    position_scale = kymograph.pixelsize[0]
    half_width_pixels = _to_half_kernel_size(track_width, position_scale)

    peaks = find_kymograph_peaks(
        kymograph_data,
        half_width_pixels,
        pixel_threshold,
        bias_correction=bias_correction,
        filter_width=filter_width / kymograph.pixelsize_um[0] if filter_width is not None else 0.5,
        adjacency_half_width=half_width_pixels if adjacency_filter else None,
        rect=_to_pixel_rect(rect, kymograph.pixelsize[0], kymograph.line_time_seconds)
        if rect
        else None,
    )

    if not peaks:
        return KymoTrackGroup([])

    # Convert algorithm parameters to pixel units
    velocity_pixels = velocity * kymograph.line_time_seconds / position_scale
    diffusion_pixels = diffusion / (position_scale**2 / kymograph.line_time_seconds)
    sigma_pixels = sigma / position_scale if sigma else half_width_pixels

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
        KymoTrack._from_centroid_estimate(
            track.time_idx,
            track.coordinate_idx,
            kymograph,
            channel,
            half_width_pixels,
            kymograph.line_time_seconds,  # Shortest observable dwell-time is given by line time
        )
        for track in tracks
    ]

    return KymoTrackGroup(tracks)


def _interp_to_frame(time_idx, coordinate):
    """Interpolate a set of time and positional coordinates back to integer frames

    Note
    ----
    Beyond the line edges, this function keeps the coordinate value constant. The reason why this
    is preferable over extrapolation is that the line detection phase can detect vertical lines.
    If such a vertical line occurs, the extrapolated value is typically highly inaccurate.

    Parameters
    ----------
    time_idx : array_like
        Time indices.
    coordinate : array_like
        Positional coordinates.

    Returns
    -------
    new_time_idx : np.ndarray
    interpolated_coordinate : np.ndarray
    """
    new_time_idx = np.arange(
        int(np.floor(time_idx[0] + 0.5)),  # consistently rounds halves down
        int(np.ceil(time_idx[-1] - 0.5)) + 1,  # consistently rounds halves up and adds one
    )
    return new_time_idx, np.interp(new_time_idx, time_idx, coordinate)


def track_lines(
    kymograph,
    channel,
    line_width,
    max_lines,
    *,
    start_threshold=0.005,
    continuation_threshold=0.005,
    angle_weight=10.0,
    rect=None,
):
    """Track particles on an image using an algorithm that looks for line-like structures.

    This function tracks particles in an image. It takes a pixel image, and traces lines on it.
    These lines are subsequently refined and/or used to extract intensities or other parameters.
    Refinement is performed using a bias-corrected centroid optimization according to [2]_.

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
    kymograph : Kymo
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

    Returns
    -------
    kymotrack_group : KymoTrackGroup

    Raises
    ------
    ValueError
        If the line_width is not larger than 3 pixels if refinement is selected.

    References
    ----------
    .. [1] Steger, C. (1998). An unbiased detector of curvilinear structures. IEEE Transactions on
           pattern analysis and machine intelligence, 20(2), 113-125.
    .. [2] Berglund, A. J., McMahon, M. D., McClelland, J. J., & Liddle, J. A. (2008).
           Fast, bias-free algorithm for tracking single particles with variable size and
           shape. Optics express, 16(18), 14064-14075.
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

    kymotrack_group = KymoTrackGroup(
        [
            KymoTrack(
                *_interp_to_frame(line.time_idx, line.coordinate_idx),
                kymograph,
                channel,
                kymograph.line_time_seconds,  # Shortest observable dwell-time is given by line time
            )
            for line in lines
        ]
    )

    return refine_tracks_centroid(kymotrack_group, track_width=line_width, bias_correction=True)


def filter_tracks(tracks, minimum_length=1, *, minimum_duration=0):
    """Remove tracks shorter than specified criteria from the list.

    This can be used to enforce a minimum number of frames a spot has to be detected in order
    to be considered a valid track.

    Parameters
    ----------
    tracks : List[KymoTrack]
        Detected tracks on a kymograph.
    minimum_length : int, optional
        Minimum number of tracked points for the track to be accepted (default: 1).
    minimum_duration : seconds, optional
        Minimum duration in seconds for a track to be accepted (default: 0).
    """

    def minimum_observable_time(track, min_length, min_duration):
        line_time = track._kymo.line_time_seconds
        minimum_length_based = (min_length - 1) * line_time

        # When we filter with a minimum duration, we lose all durations up to the next
        # full time step.
        minimum_duration_based = np.ceil(min_duration / line_time) * line_time
        return max(minimum_length_based, minimum_duration_based)

    return KymoTrackGroup(
        [
            track._with_minimum_time(
                max(
                    # We can't unfilter tracks.
                    track._minimum_observable_duration
                    if track._minimum_observable_duration is not None
                    else 0,
                    minimum_observable_time(track, minimum_length, minimum_duration),
                )
            )
            for track in tracks
            if len(track) >= minimum_length and track.duration >= minimum_duration
        ]
    )


def _apply_to_group(tracks, func, *args, **kwargs):
    """Break a group of tracks into sets with a single source kymograph, apply `func`,
    and rebuild group in original order.

    Parameters
    ----------
    tracks : KymoTrackGroup
        Tracks to apply function to
    func : callable
        Function to be applied to tracks
    *args, **kwargs
        Additional arguments supplied to `func`
    """
    groups, indices = tracks._tracks_by_kymo()
    groups = [func(group, *args, **kwargs) for group in groups]
    return KymoTrackGroup([track for _, track in sorted(zip(chain(*indices), chain(*groups)))])


def refine_tracks_centroid(tracks, track_width=None, bias_correction=True):
    """Refine the tracks based on the brightness-weighted centroid.

    This function interpolates the determined tracks (in time) and then uses the pixels in the
    vicinity of the tracks to make small adjustments to the estimated location. The refinement
    correction is computed by considering the brightness weighted centroid.

    .. note::

        The `track_width` parameter is given in physical units, but the algorithm works with
        discrete pixels. In order to avoid bias in the result, the number of pixels to use is
        rounded up to the nearest odd value.

    Parameters
    ----------
    tracks : List[KymoTrack] or KymoTrackGroup
        Detected tracks on a kymograph
    track_width : float
        Expected (spatial) spot size in physical units. Must be larger than zero.
        If `None`, the default is 0.35 (half the wavelength of the red limit of the visible
        spectrum) for kymographs calibrated in microns. For kymographs calibrated in kilobase pairs
        the corresponding value is calculated using 0.34 nm/bp (from duplex DNA).
    bias_correction : bool
        Correct coordinate bias by ensuring that the window is symmetric (see [1]_ for more
        information). Note that while this increases the variance (reduces precision), it can
        greatly reduce the bias when there is a high background.

    Returns
    -------
    refined_tracks : KymoTrackGroup
        KymoTrackGroup with refined coordinates.

    Raises
    ------
    ValueError
        If the track width is not at least 3 pixels.

    References
    ----------
    .. [1] Berglund, A. J., McMahon, M. D., McClelland, J. J., & Liddle, J. A. (2008).
           Fast, bias-free algorithm for tracking single particles with variable size and
           shape. Optics express, 16(18), 14064-14075.
    """
    tracks = KymoTrackGroup(tracks) if isinstance(tracks, (list, tuple)) else tracks

    if len(tracks._kymos) > 1:
        return _apply_to_group(
            tracks, refine_tracks_centroid, track_width=track_width, bias_correction=bias_correction
        )

    if not tracks:
        return KymoTrackGroup([])

    # the existence of tracks implies there is at least one source kymo
    # _apply_to_group ensures there is only a single source kymo at this point
    kymo = tracks._kymos[0]

    minimum_width = 3 * kymo.pixelsize[0]
    if track_width is None:
        track_width = max(_default_track_widths[kymo._calibration.unit], minimum_width)

    _validate_track_width(track_width, minimum_width, kymo._calibration.unit)

    half_width_pixels = _to_half_kernel_size(track_width, kymo.pixelsize[0])

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
        half_width_pixels,
        bias_correction=bias_correction,
    )

    track_ids = np.hstack(
        [np.full(len(track.time_idx), j) for j, track in enumerate(interpolated_tracks)]
    )

    new_tracks = [
        KymoTrack._from_centroid_estimate(
            time_idx[track_ids == j],
            coordinate_idx[track_ids == j],
            track._kymo,
            track._channel,
            half_width_pixels,
            track._minimum_observable_duration,
        )
        for j, track in enumerate(interpolated_tracks)
    ]
    return KymoTrackGroup(new_tracks)


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
    tracks : List[KymoTrack] or KymoTrackGroup
        Detected tracks on a kymograph.
    window : int
        Number of pixels on either side of the estimated track to include in the optimization data.
        The fitting window should be large enough to capture the tails of the gaussian PSF, but
        ideally small enough such that it will not include data from nearby tracks.
    refine_missing_frames : bool
        Whether to estimate location for frames which were missed in initial peak finding.
    overlap_strategy : {'multiple', 'ignore', 'skip'}
        How to deal with frames in which the fitting window of two
        :class:`~lumicks.pylake.kymotracker.kymotrack.KymoTrack`'s overlap.

        - 'multiple' : fit the peaks simultaneously (deprecated, use `simultaneous` instead).
        - 'simultaneous' : fit the peaks simultaneously with estimation bounds between peaks.
        - 'ignore' : do nothing, fit the frame as-is (ignoring overlaps).
        - 'skip' : skip optimization of the frame; remove from returned :class:`~lumicks.pylake.kymotracker.kymotrack.KymoTrack`.
    initial_sigma : float
        Initial guess for the `sigma` parameter.
    fixed_background : float
        Fixed background parameter in photons per second.
        When supplied, the background is not estimated but fixed at this value.
    """
    if overlap_strategy not in ("ignore", "skip", "multiple", "simultaneous"):
        raise ValueError("Invalid overlap strategy selected.")

    if overlap_strategy == "multiple":
        warnings.warn(
            DeprecationWarning(
                'overlap_strategy="multiple" is deprecated. Use "simultaneous" instead. The '
                'strategy "simultaneous" enforces optimization bounds between the peak positions. '
                "This helps prevent the refinement procedure from reassigning points to the wrong "
                "track when a track momentarily disappears. When using the overlap strategy "
                '"multiple" individual overlapping Gaussians could switch positions leading to'
                "spurious track crossings."
            )
        )

    if len(tracks._kymos) > 1:
        return _apply_to_group(
            tracks,
            refine_tracks_gaussian,
            window,
            refine_missing_frames,
            overlap_strategy,
            initial_sigma=initial_sigma,
            fixed_background=fixed_background,
        )

    if not tracks:
        return KymoTrackGroup([])

    if refine_missing_frames:
        tracks = KymoTrackGroup([track.interpolate() for track in tracks])

    # the existence of tracks implies there is at least one source kymo
    # _apply_to_group ensures there is only a single source kymo at this point
    kymo = tracks._kymos[0]
    channel = tracks._channel
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
                enforce_position_bounds=overlap_strategy == "simultaneous",
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
            KymoTrack(
                t,
                GaussianLocalizationModel(*np.vstack(p).T),
                kymo,
                channel,
                ref_track._minimum_observable_duration,
            )
            for t, p, ref_track in zip(refined_tracks_time_idx, refined_tracks_parameters, tracks)
            if len(t) > 0
        ]
    )
