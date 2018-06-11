import enum
import math
import numpy as np


def reconstruct_image(data, infowave, pixels_per_line, reduce=np.sum):
    """Reconstruct a scan or kymograph image from raw data

    Parameters
    ----------
    data : array_like
        Raw data to use for the reconstruction. E.g. photon counts or force samples.
    infowave : array_like
        The famous infowave.
    pixels_per_line : int
        The number of pixels on the fast axis of the scan.
    reduce : callable
        A function which reduces multiple sample into a pixel. Usually `np.sum`
        for photon counts and `np.mean` for force samples.

    Returns
    -------
    np.ndarray
    """
    assert data.size == infowave.size

    class Code(enum.IntEnum):
        discard = 0
        use = 1
        pixel_boundary = 2

    # Example infowave:
    #  1 0 0 1 0 1 2 0 1 0 0 1 0 1 0 1 0 0 1 0 2 0 1 0 1 0 0 1 0 1 0 1 2 1 0 0 1
    #              ^ <-----------------------> ^                       ^
    #                       one pixel
    valid_idx = infowave != Code.discard
    infowave = infowave[valid_idx]

    # After discard:
    #  1 1 1 2 1 1 1 1 1 2 1 1 1 1 1 2 1 1 1
    #        ^ <-------> ^           ^
    #         pixel_size (i.e. data samples per pixel)
    pixel_sizes = np.diff(np.flatnonzero(infowave == Code.pixel_boundary))
    pixel_size = pixel_sizes[0]
    # For now we assume that every pixel consists of the same number of samples
    assert np.all(pixel_sizes == pixel_size)

    def round_up(size, n):
        """Round up `size` to the nearest multiple of `n`"""
        return int(math.ceil(size / n)) * n

    data = data[valid_idx]
    data.resize(round_up(data.size, pixel_size))

    pixels = reduce(data.reshape(-1, pixel_size), axis=1)
    pixels.resize(round_up(pixels.size, pixels_per_line))
    return pixels.reshape(-1, pixels_per_line)
