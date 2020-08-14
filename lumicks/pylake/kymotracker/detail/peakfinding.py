import numpy as np
from scipy.ndimage import gaussian_filter, grey_dilation
from scipy.signal import convolve2d


def peak_estimate(data, dilation_factor, thresh):
    """Estimate initial peak locations from data. Peaks are detected by dilating the image, and then determining which
    pixels did not change. These pixels correspond to local maxima. A threshold is then applied to select which ones
    are relevant.

    Parameters
    ----------
    data : array_like
        A 2D image of pixel data.
    dilation_factor : int
        How much to dilate the image.
    thresh : float
        Threshold for accepting something as a peak.
    """
    data = gaussian_filter(data.astype(np.float64), [.5, .5])
    dilated = grey_dilation(data, (dilation_factor, 0))
    dilated[dilated < thresh] = -1
    position, time_points = np.where(data == dilated)
    return position, time_points


def refine_peak_based_on_moment(data, position, time_points, kernel_size, max_iter=100, eps=1e-7):
    """This function adjusts the position estimate by a brightness weighted centroid around the initial estimate. This
    estimate is obtained by filtering the image with a kernel. If a pixel offset has a larger magnitude than 0.5 then
    the pixel is moved and the centroid recomputed. The process is repeated until there are no more changes. Convergence
    usually occurs within a few iterations.

    Parameters
    ----------
    data : array_like
        A 2D image of pixel data (first axis corresponds to positions, second to time points).
    position : array_like
        Initial position estimates.
    time_points : array_like
        Time points at which the position estimates were made.
    kernel_size : int
        Refinement kernel size.
    max_iter : int
        Maximum number of iterations
    eps : float
        We add a little offset to the normalization to prevent divisions by zeros on pixels that did not have any photon
        counts. Eps sets this offset.
    """
    dir_kernel = np.expand_dims(np.arange(kernel_size, -(kernel_size + 1), -1), 1)
    mean_kernel = np.ones((2 * kernel_size + 1, 1))
    position = np.copy(position)

    m0 = convolve2d(data, mean_kernel, 'same')
    subpixel_offset = convolve2d(data, dir_kernel, 'same') / (m0 + eps)

    iteration = 0
    done = False
    max_position = subpixel_offset.shape[0]
    while not done:
        offsets = subpixel_offset[position, time_points]
        out_of_bounds, = np.nonzero(abs(offsets) > 0.5)
        position[out_of_bounds] += np.sign(offsets[out_of_bounds]).astype(np.int)

        # Edge cases (literally)
        low = position < 0
        position[low] = 0
        high = position >= max_position
        position[high] = max_position - 1

        done = out_of_bounds.size - np.sum(low) - np.sum(high) == 0

        if iteration > max_iter:
            raise RuntimeError("Iteration limit exceeded")

        iteration += 1

    return position + subpixel_offset[position, time_points], time_points
