import numpy as np
from scipy.ndimage import gaussian_filter, grey_dilation
from scipy.signal import convolve2d
import math


def peak_estimate(data, half_width, thresh):
    """Estimate initial peak locations from data.

    Peaks are detected by dilating the image, and then determining which pixels did not change. These pixels correspond
    to local maxima. A threshold is then applied to select which ones are relevant.

    Parameters
    ----------
    data : array_like
        A 2D image of pixel data.
    half_width : int
        How much to dilate the image in pixels. This is value should be half of the width we are looking for
        (rounded upwards). Prior to peak finding, the image is dilated symmetrically. With a half_width of 1 this
        means turning [0 0 1 0 0] into [0 1 1 1 0] prior to peak-finding.
    thresh : float
        Threshold for accepting something as a peak.
    """
    dilation_factor = int(math.ceil(half_width)) * 2 + 1
    data = gaussian_filter(data, [0.5, 0])
    dilated = grey_dilation(data, (dilation_factor, 0))
    dilated[dilated < thresh] = -1
    coordinates, time_points = np.where(data == dilated)
    return coordinates, time_points


class KymoPeaks:
    """Stores local maxima found in a kymograph on a per-frame basis."""

    class Frame:
        """Stores local maxima found in a kymograph for a single frame."""

        def __init__(self, coordinates, time_points, peak_amplitudes):
            self.coordinates = coordinates
            self.time_points = time_points
            self.peak_amplitudes = peak_amplitudes
            self.unassigned = []

        def reset_assignment(self):
            self.unassigned = np.ones(self.time_points.shape, dtype=bool)

    def __init__(self, coordinates, time_points, peak_amplitudes):
        """Kymograph peaks

        Parameters
        ----------
        coordinates : np.ndarray
            Positional coordinates of detected peaks
        time_points : np.ndarray
            Time points (in frame indices) of detected peaks
        peak_amplitudes : np.ndarray
            Peak amplitudes of detected peaks

        Raises
        ------
        ValueError
            When no points are given or when `coordinates`, `time_points` and `peak_amplitudes`
            don't have the same number of elements.
        """
        if len(time_points) == 0:
            raise ValueError("You need to provide at least one time point")

        if any(len(time_points) != len(x) for x in (coordinates, peak_amplitudes)):
            raise ValueError(
                f"Number of time points ({len(time_points)}), coordinates ({len(coordinates)}) and "
                f"peak amplitudes ({len(peak_amplitudes)}) must be equal"
            )

        self.frames = []
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


def bounds_to_centroid_data(left_edge, right_edge):
    """Helper function to return selection indices, pixel centers and weights.

    This function generates indices for sampling based on the left-most and right-most
    bound. It also generates appropriate pixel centers and weights that account for
    the fact that they are only partial pixels.

    Parameters
    ----------
    left_edge, right_edge : float
        Lower and upper pixel edge

    Returns
    -------
    selection : np.ndarray
        Indices which select which points to use.
    centers : np.ndarray
        Pixel centers, where the edge pixels are corrected for being partial.
    weights : np.ndarray
        Weights that account for down-weighting the edges.
    """
    selection = np.arange(left_edge, np.ceil(right_edge), dtype=int)
    centers = selection + 0.5
    weights = np.ones(selection.size)
    centers[0] = (left_edge + np.floor(left_edge) + 1) / 2
    centers[-1] = (right_edge + np.ceil(right_edge) - 1) / 2
    weights[0] = 1.0 - (left_edge - np.floor(left_edge))
    weights[-1] = 1.0 - (np.ceil(right_edge) - right_edge)

    return selection, centers, weights


def unbiased_centroid(data, tolerance=1e-3, max_iterations=50, epsilon=1e-8):
    """Perform an unbiased centroid estimation

    The bias in centroid refinement is proportional to the asymmetry around the spot. To remove this
    bias, we define a virtual window that's symmetric around the spot position. In practice, this
    involves removing fractional pixels at the edges. If we want to shift the center of the image
    by 1 pixel, we need to move an edge point 2 pixels.

    Fractional pixels are dealt with by down-weighting edge pixels and recomputing a new pixel
    center for those pixels.

    For more information on this method, please refer to [1]_.

    Parameters
    ----------
    data : np.ndarray
        1D array
    tolerance : float
        Convergence tolerance
    max_iterations : int
        Maximum iterations
    epsilon : float
        Small value to make sure we don't divide by zero

    Returns
    -------
    estimated_position : float
        Estimated subpixel position of the spot after bias correction. The origin (position 0) of
        the position is defined at the window origin.

    References
    ----------
    .. [1] Berglund, A. J., McMahon, M. D., McClelland, J. J., & Liddle, J. A. (2008).
           Fast, bias-free algorithm for tracking single particles with variable size and
           shape. Optics express, 16(18), 14064-14075.
    """
    lb = 0
    ub = data.size

    last_position = None
    for _ in range(max_iterations):
        # Too few pixels left to work with. Just return the last best guess or if none exists, the
        # average.
        if ub - lb < 2:
            return last_position if last_position is not None else (lb + ub) / 2

        selection, centers, weights = bounds_to_centroid_data(lb, ub)

        range_center = (lb + ub) / 2
        chunk = data[selection]
        weighted_coord = np.sum(weights * centers * chunk)
        weighted_sum = np.sum(weights * chunk)

        if weighted_sum < epsilon:
            # No photons here! Best guess is center of the range.
            return range_center - 0.5

        est_position = weighted_coord / weighted_sum

        if est_position > range_center:
            lb += 2 * (est_position - range_center)
        else:
            ub += 2 * (est_position - range_center)

        if last_position and np.abs(est_position - last_position) < tolerance:
            break

        last_position = est_position

    # Internally, we used a coordinate system that has pixel centers at 0.5, 1.5 etc. This is
    # practical because it simplifies the code. Externally, we define pixel centers at 0, 1, 2 etc,
    # so we subtract a half here.
    return est_position - 0.5


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

    half_kernel_size = int(math.ceil(half_kernel_size))
    dir_kernel = np.expand_dims(np.arange(half_kernel_size, -(half_kernel_size + 1), -1), 1)
    mean_kernel = np.ones((2 * half_kernel_size + 1, 1))
    coordinates = np.copy(coordinates)

    m0 = convolve2d(data, mean_kernel, "same")
    subpixel_offset = convolve2d(data, dir_kernel, "same") / (m0 + eps)

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
        data = np.copy(data)  # Our slicing operation is not allowed on a memoryview
        output_coords = np.zeros(output_coords.size)
        for idx, (coordinate, time_point) in enumerate(zip(coordinates, time_points)):
            centroid_estimate = unbiased_centroid(
                data[coordinate - half_kernel_size : coordinate + half_kernel_size + 1, time_point]
            )
            output_coords[idx] = coordinate + centroid_estimate - half_kernel_size

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
