import numpy as np
from scipy.ndimage import gaussian_filter, grey_dilation
from scipy.signal import convolve2d
import math


def peak_estimate(data, half_width, thresh):
    """Estimate initial peak locations from data. Peaks are detected by dilating the image, and then determining which
    pixels did not change. These pixels correspond to local maxima. A threshold is then applied to select which ones
    are relevant.

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
    data = gaussian_filter(data.astype(np.float64), [.5, .5])
    dilated = grey_dilation(data, (dilation_factor, 0))
    dilated[dilated < thresh] = -1
    coordinates, time_points = np.where(data == dilated)
    return coordinates, time_points


def refine_peak_based_on_moment(data, coordinates, time_points, half_kernel_size, max_iter=100, eps=1e-7):
    """This function adjusts the coordinates estimate by a brightness weighted centroid around the initial estimate.
    This estimate is obtained by filtering the image with a kernel. If a pixel offset has a larger magnitude than 0.5
    then the pixel is moved and the centroid recomputed. The process is repeated until there are no more changes.
    Convergence usually occurs within a few iterations.

    Parameters
    ----------
    data : array_like
        A 2D image of pixel data (first axis corresponds to coordinates, second to time points).
    coordinates : array_like
        Initial coordinate estimates.
    time_points : array_like
        Time points at which the coordinate estimates were made.
    half_kernel_size : int
        Half of the kernel size in pixels. The kernel is used to refine the line estimate. The kernel size used for this
        refinement will be 2 * half_kernel_size + 1.
    max_iter : int
        Maximum number of iterations
    eps : float
        We add a little offset to the normalization to prevent divisions by zeros on pixels that did not have any photon
        counts. Eps sets this offset.
    """
    half_kernel_size = int(math.ceil(half_kernel_size))
    dir_kernel = np.expand_dims(np.arange(half_kernel_size, -(half_kernel_size + 1), -1), 1)
    mean_kernel = np.ones((2 * half_kernel_size + 1, 1))
    coordinates = np.copy(coordinates)

    m0 = convolve2d(data, mean_kernel, 'same')
    subpixel_offset = convolve2d(data, dir_kernel, 'same') / (m0 + eps)

    iteration = 0
    done = False
    max_coordinates = subpixel_offset.shape[0]
    while not done:
        offsets = subpixel_offset[coordinates, time_points]
        out_of_bounds, = np.nonzero(abs(offsets) > 0.5)
        coordinates[out_of_bounds] += np.sign(offsets[out_of_bounds]).astype(np.int)

        # Edge cases (literally)
        low = coordinates < 0
        coordinates[low] = 0
        high = coordinates >= max_coordinates
        coordinates[high] = max_coordinates - 1

        done = out_of_bounds.size - np.sum(low) - np.sum(high) == 0

        if iteration > max_iter:
            raise RuntimeError("Iteration limit exceeded")

        iteration += 1

    return coordinates + subpixel_offset[coordinates, time_points], time_points, m0[coordinates, time_points]


def merge_close_peaks(coordinates, time_points, peak_amplitude, minimum_distance):
    """Merge peaks that are too close to each-other vertically. Peaks that fall within the dilation mask are spurious
    and likely not peaks we want. When two peaks fall below the minimum distance, the smallest one will be discarded.

    Parameters:
    ----------
    coordinates, time_points, peak_amplitude : array_like
    minimum_distance : int
        Minimum distance between peaks to enforce
    """
    assert len(coordinates) == len(time_points)
    assert len(peak_amplitude) == len(time_points)

    mask = np.ones(coordinates.shape, dtype=bool)
    max_frame = math.ceil(np.max(time_points))
    for current_frame in np.arange(max_frame + 1):
        in_frame_idx, = np.where(np.logical_and(time_points >= current_frame, time_points < current_frame + 1))
        coordinate_difference = np.diff(np.sort(coordinates[in_frame_idx]))
        too_close, = np.where(coordinate_difference < minimum_distance)
        too_close = in_frame_idx[too_close]

        # If the right peak of the two candidates is smaller than the left one, take that one for removal instead
        right_lower = peak_amplitude[too_close + 1] < peak_amplitude[too_close]
        too_close[right_lower] += 1
        mask[too_close] = False

    return coordinates[mask], time_points[mask], peak_amplitude[mask]
