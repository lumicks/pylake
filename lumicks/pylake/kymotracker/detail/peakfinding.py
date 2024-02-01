import math
from typing import Optional
from dataclasses import dataclass

import numpy as np
import scipy


def find_kymograph_peaks(
    kymograph_data,
    half_width_pixels,
    threshold,
    bias_correction=True,
    rect=None,
    filter_width=0.5,
    adjacency_half_width=None,
):
    """Find local peaks in a kymograph.

    Parameters
    ----------
    kymograph_data : np.ndarray
        Raw single channel kymograph image data.
    half_width_pixels : int
        Half width in pixels. The kernel size used in refinement will be 2 * half_width + 1
    threshold : float
        Threshold to use.
    bias_correction : bool, optional
        Enable bias correction when performing the refinement. Default: True.
    rect : tuple, optional
        Rectangle to crop results to. Tuple of integer pixel values referring to a pair of
        (time, position) coordinates. Default: no cropping.
    filter_width : float
        Width of the point spread function in pixels. Default: 0.5.
    adjacency_half_width : int, optional
        When provided, a detection always needs a detection within a certain cutoff radius (in
        pixels) in an adjacent frame. This can be used to suppress singular noise peaks.

        If the center pixel is being tested with an adjacency_half_width of 1:

          [0 0 0]           [1 0 0]              [0 0 0]
          [1 0 0]           [0 0 0]              [0 0 0]
          [0 1 0] => Pass   [0 1 0] => Reject    [0 1 0] ==> Pass
          [0 0 0]           [0 0 0]   (too far)  [0 0 1]
          [0 0 0]           [0 0 0]              [0 0 0]
    """
    filtered_data = scipy.ndimage.gaussian_filter(kymograph_data, [filter_width, 0], output=float)
    coordinates, time_points = peak_estimate(
        filtered_data, half_width_pixels, threshold, adjacency_half_width
    )
    if len(coordinates) == 0:
        return KymoPeaks([], [], [])

    position, time, m0 = refine_peak_based_on_moment(
        kymograph_data,
        coordinates,
        time_points,
        half_width_pixels,
        bias_correction=bias_correction,
    )

    if rect:
        (t0, p0), (t1, p1) = rect
        mask = (position >= p0) & (position < p1) & (time >= t0) & (time < t1)
        position, time, m0 = position[mask], time[mask], m0[mask]

        if len(position) == 0:
            return KymoPeaks([], [], [])

    return merge_close_peaks(KymoPeaks(position, time, m0), half_width_pixels)


def peak_estimate(data, half_width, thresh, adjacency_half_width=None):
    """Estimate initial peak locations from data.

    Peaks are detected by dilating the image, and then determining which pixels did not change.
    These pixels correspond to local maxima. A threshold is then applied to select which ones are
    relevant.

    Parameters
    ----------
    data : array_like
        A 2D image of pixel data.
    half_width : int
        How much to dilate the image in pixels. This is value should be half of the width we are
        looking for (rounded upwards). Prior to peak finding, the image is dilated symmetrically.
        With a half_width of 1 this means turning [0 0 1 0 0] into [0 1 1 1 0] prior to
        peak-finding.
    thresh : float
        Threshold for accepting something as a peak.
    adjacency_half_width : int, optional
        When provided, a detection always needs a detection within a certain cutoff radius in an
        adjacent frame. This can be used to suppress singular noise peaks.
    """
    import scipy.signal

    if thresh <= (minimum_value := np.min(data)):
        raise RuntimeError(
            f"Threshold ({thresh}) cannot be lower than or equal to the lowest filtered "
            f"pixel ({minimum_value})"
        )

    dilation_factor = int(math.ceil(half_width)) * 2 + 1
    dilated = scipy.ndimage.grey_dilation(data, (dilation_factor, 0))

    thresholded_local_maxima = np.logical_and(data == dilated, data >= thresh)

    # Reject maxima that have no maximum in an adjacent frame.
    if adjacency_half_width:
        mask_size = 2 * adjacency_half_width + 1
        mask = np.vstack((np.ones(mask_size), np.zeros(mask_size), np.ones(mask_size))).T
        thresholded_local_maxima *= (
            scipy.signal.convolve2d(thresholded_local_maxima, mask, mode="same") > 0
        )

    coordinates, time_points = np.where(thresholded_local_maxima)
    return coordinates, time_points


class KymoPeaks:
    """Stores local maxima found in a kymograph on a per-frame basis."""

    @dataclass
    class Frame:
        """Stores local maxima found in a kymograph for a single frame."""

        coordinates: np.ndarray
        time_points: np.ndarray
        peak_amplitudes: np.ndarray
        unassigned: Optional[np.ndarray] = None

        def __post_init__(self):
            if not self.unassigned:
                self.reset_assignment()

        def reset_assignment(self):
            self.unassigned = np.ones(self.time_points.shape, dtype=bool)

    def __init__(self, coordinates, time_points, peak_amplitudes):
        """Kymograph peaks

        Parameters
        ----------
        coordinates : array_like
            Positional coordinates of detected peaks
        time_points : array_like
            Time points (in frame indices) of detected peaks
        peak_amplitudes : array_like
            Peak amplitudes of detected peaks

        Raises
        ------
        ValueError
            When no points are given or when `coordinates`, `time_points` and `peak_amplitudes`
            don't have the same number of elements.
        """
        if any(len(time_points) != len(x) for x in (coordinates, peak_amplitudes)):
            raise ValueError(
                f"Number of time points ({len(time_points)}), coordinates ({len(coordinates)}) and "
                f"peak amplitudes ({len(peak_amplitudes)}) must be equal"
            )

        self.frames = []
        coordinates, time_points, peak_amplitudes = (
            np.asarray(c) for c in (coordinates, time_points, peak_amplitudes)
        )

        if len(time_points) == 0:
            return

        max_frame = math.ceil(np.max(time_points))
        for current_frame in np.arange(max_frame + 1):
            (in_frame_idx,) = np.where(
                np.logical_and(time_points >= current_frame, time_points < current_frame + 1)
            )
            self.frames.append(
                self.Frame(
                    coordinates[in_frame_idx],
                    time_points[in_frame_idx],
                    peak_amplitudes[in_frame_idx],
                )
            )

    def reset_assignment(self):
        for frame in self.frames:
            frame.reset_assignment()

    def flatten(self):
        coordinates = np.hstack([frame.coordinates for frame in self.frames])
        time_points = np.hstack([frame.time_points for frame in self.frames])
        peak_amplitudes = np.hstack([frame.peak_amplitudes for frame in self.frames])

        return coordinates, time_points, peak_amplitudes

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.frames[index]
        else:
            raise IndexError("Only integer indexing is allowed.")

    def __repr__(self):
        return f"KymoPeaks(N={len(self.frames)})"

    def __bool__(self):
        return bool(self.frames)


def bounds_to_centroid_data(index_array, left_edges, right_edges):
    """Helper function to return selection indices, pixel centers and weights.

    This function generates indices for sampling based on the left-most and right-most bound. It
    also generates appropriate pixel centers and weights that account for the fact that they are
    only partial pixels.

    Parameters
    ----------
    index_array: np.ndarray
        2D pixel indices with shape `(N, M)` where `N` is the number of pixels being refined and
        `M` is the window size used for refinement.
    left_edges, right_edges : np.ndarray
        Lower and upper pixel edges

    Returns
    -------
    selection : np.ndarray
        Boolean mask with shape `(N, M)` which select which points from the window to use.
        `N` is the number of pixels being refined, while `M` is the window size used for refinement.
    centers : np.ndarray
        Pixel center positions with shape `(N, M)` with edge pixel positions accounting for being
        partial pixels. `N` is the number of pixels being refined, while `M` is the window size
        used for refinement.
    weights : np.ndarray
        Weights with shape `(N, M)` that account for down-weighting the edges.
        `N` is the number of pixels being refined, while `M` is the window size used for refinement.
    """
    left_floor = np.floor(left_edges).astype(int)
    right_ceil = np.ceil(right_edges).astype(int)

    # Upper bound needs to be rounded inwards
    right_floor = np.floor(np.nextafter(right_edges, 0)).astype(int)

    # Find which pixels we are still using
    active = np.logical_and(
        index_array >= left_floor[:, np.newaxis], index_array < right_ceil[:, np.newaxis]
    )

    num_points = left_floor.size
    centers = active * (index_array + 0.5)

    centers[np.arange(num_points), left_floor] = (left_edges + left_floor + 1) / 2
    centers[np.arange(num_points), right_floor] = (right_edges + right_ceil - 1) / 2

    # Pixels outside the bounds are not used
    weights = active.astype(float)

    # Fractional pixels are down-weighted.
    weights[np.arange(num_points), left_floor] = 1.0 - (left_edges - np.floor(left_edges))
    weights[np.arange(num_points), right_floor] = 1.0 - (np.ceil(right_edges) - right_edges)
    return active, centers, weights


def unbiased_centroid(position, data, tolerance=1e-3, max_iterations=50, epsilon=1e-8):
    """Perform an unbiased centroid estimation in a vectorized manner

    The bias in centroid refinement is proportional to the asymmetry around the spot. To remove this
    bias, we define a virtual window that's symmetric around the spot position. In practice, this
    involves removing fractional pixels at the edges. If we want to shift the center of the image
    by 1 pixel, we need to move an edge point 2 pixels.

    Fractional pixels are dealt with by down-weighting edge pixels and recomputing a new pixel
    center for those pixels.

    For more information on this method, please refer to [1]_.

    Parameters
    ----------
    position : np.ndarray
        Subpixel coordinates obtained with regular centroid method.
    data : np.ndarray
        2D image array with data used for refinement. It contains a small region around each spot.
        It has shape `(N, M)`, with `N` the number of pixels being refined, while `M` is the
        window size used for refinement.
    tolerance : float
        Convergence tolerance
    max_iterations : int
        Maximum iterations
    epsilon : float
        Small value to make sure we don't divide by zero

    Returns
    -------
    estimated_positions : np.ndarray
        Estimated subpixel positions of the spot after bias correction. The origin (position 0) of
        the position is defined at the window origin.

    References
    ----------
    .. [1] Berglund, A. J., McMahon, M. D., McClelland, J. J., & Liddle, J. A. (2008).
           Fast, bias-free algorithm for tracking single particles with variable size and
           shape. Optics express, 16(18), 14064-14075.
    """
    lb = np.tile(0.0, data.shape[0])
    ub = np.tile(float(data.shape[1]), data.shape[0])
    index_array = np.tile(np.arange(data.shape[1]), (data.shape[0], 1))

    # In this function we define the internal coordinate system with the origin at the pixel edge
    last_position = position + 0.5
    for _ in range(max_iterations):
        selection, centers, weights = bounds_to_centroid_data(index_array, lb, ub)

        range_center = (lb + ub) / 2.0
        weights *= data
        weighted_coord = np.sum(weights * centers, axis=1)
        weighted_sum = np.sum(weights, axis=1)
        empty_pixels = weighted_sum < epsilon
        weighted_sum[empty_pixels] = 1.0  # We are not using these anyway

        est_position = weighted_coord / weighted_sum

        # Only consider those pixels where we had sufficient data
        sufficient_data = np.logical_and(ub - lb > 2, np.logical_not(empty_pixels))
        position[sufficient_data] = est_position[sufficient_data]

        move_lb = position > range_center
        lb[move_lb] = lb[move_lb] + 2.0 * (position[move_lb] - range_center[move_lb])
        move_ub = np.logical_not(move_lb)
        ub[move_ub] = ub[move_ub] + 2.0 * (position[move_ub] - range_center[move_ub])

        if np.max(np.abs(position - last_position)) < tolerance:
            break

        # We change position in place each iteration, so we need to be explicit about the copy
        last_position = position.copy()

    # Internally, we used a coordinate system that has pixel centers at 0.5, 1.5 etc. This is
    # practical because it simplifies the code. Externally, we define pixel centers at 0, 1, 2 etc,
    # so we subtract a half here.
    return position - 0.5


def _clip_kernel_to_edge(max_half_width, coordinates, dataset_width):
    """Find the number of pixels between the coordinates and the image edge.

    The kernel half width is defined as [center - hw : center + hw + 1].

    Parameters
    ----------
    max_half_width : int
        Desired kernel size. If the kernel completely fits, then this is the returned value.
    coordinates : np.ndarray
        1D array of coordinates
    dataset_width : int
        Shape of the data matrix to clip against (basically the maximum value for the coordinate)

    Returns
    -------
    maximum_half_width : np.ndarray
        Array containing number of pixels between given coordinates and the window edge.
    """
    return np.minimum(max_half_width, np.minimum(coordinates, dataset_width - coordinates - 1))


def refine_peak_based_on_moment(
    data, coordinates, time_points, half_kernel_size, max_iter=100, eps=1e-7, bias_correction=True
):
    """This function adjusts the coordinates estimate by a brightness weighted centroid around the
    initial estimate. This estimate is obtained by filtering the image with a kernel. If a pixel
    offset has a larger magnitude than 0.5 then the pixel is moved and the centroid recomputed. The
    process is repeated until there are no more changes. Convergence usually occurs within a few
    iterations.

    Parameters
    ----------
    data : array_like
        A 2D image of pixel data (first axis corresponds to coordinates, second to time points).
    coordinates : array_like
        Initial coordinate estimates.
    time_points : array_like
        Time points at which the coordinate estimates were made.
    half_kernel_size : int
        Half of the kernel size in pixels. The kernel is used to refine the line estimate. The
        kernel size used for this refinement will be 2 * half_kernel_size + 1.
    max_iter : int
        Maximum number of iterations
    eps : float
        We add a little offset to the normalization to prevent divisions by zeros on pixels that
        did not have any photon counts. Eps sets this offset.
    bias_correction : bool
        Whether to apply a bias correction at the end of the centroid refinement.

    Returns
    -------
    output_coords : np.ndarray
        Refined output coordinates.
    time_points : np.ndarray
        Time points.
    m0 : np.ndarray
        Sum over the window.
    """
    if half_kernel_size < 1:
        raise ValueError("half_kernel_size may not be smaller than 1")

    data = np.asarray(data)
    half_kernel_size = int(math.ceil(half_kernel_size))
    dir_kernel = np.expand_dims(np.arange(half_kernel_size, -(half_kernel_size + 1), -1), 1)
    mean_kernel = np.ones((2 * half_kernel_size + 1, 1))
    coordinates = np.copy(coordinates)

    m0 = scipy.signal.convolve2d(data, mean_kernel, "same")
    subpixel_offset = scipy.signal.convolve2d(data, dir_kernel, "same") / (m0 + eps)

    max_coordinates = subpixel_offset.shape[0]
    for _ in range(max_iter):
        offsets = subpixel_offset[coordinates, time_points]
        (out_of_bounds,) = np.nonzero(abs(offsets) > 0.5)
        coordinates[out_of_bounds] += np.sign(offsets[out_of_bounds]).astype(int)

        # Edge cases (literally)
        low = coordinates < 0
        coordinates[low] = 0
        high = coordinates >= max_coordinates
        coordinates[high] = max_coordinates - 1

        if out_of_bounds.size - np.sum(low) - np.sum(high) == 0:
            break
    else:
        raise RuntimeError("Iteration limit exceeded")

    output_coords = coordinates + subpixel_offset[coordinates, time_points]

    # We found the rough location, time to refine and debias
    if bias_correction:
        data = np.copy(data)  # Our slicing operation is not allowed on a memory-view

        # We have to ensure that the full window fits, otherwise we get a big bias with this method
        half_widths = _clip_kernel_to_edge(half_kernel_size, coordinates, data.shape[0])

        # First do the ones where we actually have a full kernel
        for hw in range(1, half_kernel_size + 1):
            to_process = half_widths == hw

            if np.any(to_process):
                tp, coords = time_points[to_process], coordinates[to_process]
                init_guess = subpixel_offset[coords, tp]

                # This is kind of like data[coords - hw : coords + hw + 1, :]
                selection = (
                    np.tile(np.arange(-hw, hw + 1), (tp.shape[0], 1)) + coords[:, np.newaxis]
                )
                selected_data = data[selection, tp[:, np.newaxis]]

                centroid_estimate = unbiased_centroid(init_guess, selected_data)
                output_coords[to_process] = coords + centroid_estimate - hw

    return (
        output_coords,
        time_points,
        m0[coordinates, time_points],
    )


def merge_close_peaks(peaks, minimum_distance):
    """Merge peaks that are too close to each-other vertically.

    Peaks that fall within the dilation mask are spurious and likely not peaks we want. When two peaks fall below the
    minimum distance, the smallest one will be discarded.

    Parameters:
    ----------
    peaks : KymoPeaks
        Data structure which contains coordinates, time_points, peak_amplitude on a per frame basis.
    minimum_distance : int
        Minimum distance between peaks to enforce
    """
    for current_frame in peaks.frames:
        # Sort frame indices by the coordinate
        sort_order = np.argsort(current_frame.coordinates)
        coordinates, peak_amplitudes = (
            current_frame.coordinates[sort_order],
            current_frame.peak_amplitudes[sort_order],
        )

        coordinate_difference = np.diff(coordinates)
        (too_close,) = np.where(np.abs(coordinate_difference) < minimum_distance)

        # If the right peak of the two candidates is smaller than the left one, take that one for removal instead
        right_lower = peak_amplitudes[too_close + 1] < peak_amplitudes[too_close]
        too_close[right_lower] += 1

        mask = np.ones(current_frame.coordinates.shape, dtype=bool)
        mask[sort_order[too_close]] = False  # too_close was in sorted ordering.

        current_frame.coordinates = current_frame.coordinates[mask]
        current_frame.peak_amplitudes = current_frame.peak_amplitudes[mask]
        current_frame.time_points = current_frame.time_points[mask]

    return peaks


def _sum_track_signal(
    image, num_pixels, time_idx, coordinate_idx, reduce=np.sum, correct_origin=None
):
    """Sum pixels intensities around the track

    This function samples data from the image based on the provided coordinates. It samples a
    symmetric window of size `2 * num_pixels + 1` centered on the provided coordinates.

    Parameters
    ----------
    num_pixels : int
        Number of pixels in either direction to include in the sample.
    time_idx : arraylike
        Time coordinates.
    coordinate_idx : arraylike
        Positional coordinates.
    reduce : callable
        Function evaluated on the sample. (Default: np.sum which produces sum of photon counts).
    correct_origin : bool, optional
        Use the correct pixel origin when sampling from image. Kymotracks are defined with the
        origin of each image pixel defined at the center. Earlier versions of the method that
        samples photon counts around the track had a bug which assumed the origin at the edge
        of the pixel. Setting this flag to `True` produces the correct behavior. The default is
        set to `None` which reproduces the old behavior and results in a warning, while `False`
        reproduces the old behavior without a warning.
    """
    # Time and coordinates are being cast to an integer since we use them to index into a data
    # array. Note that coordinate pixel centers are defined at integer coordinates.
    offset = 0.0 if not correct_origin else 0.5
    return [
        reduce(
            image[max(int(c + offset) - num_pixels, 0) : int(c + offset) + num_pixels + 1, int(t)]
        )
        for t, c in zip(time_idx, coordinate_idx)
    ]
