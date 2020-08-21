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
    data = gaussian_filter(data, [.5, .5])
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
            self.unassigned = np.ones(self.time_points.shape, dtype=np.bool)

    def __init__(self, coordinates, time_points, peak_amplitudes):
        assert len(coordinates) == len(time_points)
        assert len(peak_amplitudes) == len(time_points)

        self.frames = []
        max_frame = math.ceil(np.max(time_points))
        for current_frame in np.arange(max_frame + 1):
            (in_frame_idx,) = np.where(
                np.logical_and(
                    time_points >= current_frame, time_points < current_frame + 1
                )
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

    return KymoPeaks(coordinates + subpixel_offset[coordinates, time_points], time_points, m0[coordinates, time_points])
