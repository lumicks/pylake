import enum
import math
import numpy as np


class InfowaveCode(enum.IntEnum):
    discard = 0  # this data sample does not contain useful information
    use = 1  # useful data to be used to form a pixel
    pixel_boundary = 2  # useful data and marks the last data sample in a pixel


def reconstruct_num_frames(infowave, pixels_per_line, lines_per_frame):
    """Reconstruct the number of frames in a continuous scan

    Unfortunately, continuous scans do not store the number of frames as part of their
    metadata (instead the number is zero) so we must reconstruct this number based on
    the infowave.

    Parameters
    ----------
    infowave : array_like
        The famous infowave.
    pixels_per_line : int
        The number of pixels on the fast axis of the scan.
    lines_per_frame : int
        The number of pixels on the slow axis of the scan. Only needed for multi-frame scans.

    Returns
    -------
    int
    """
    num_pixels = np.count_nonzero(infowave == InfowaveCode.pixel_boundary)
    pixels_per_frames = pixels_per_line * lines_per_frame
    return math.ceil(num_pixels / pixels_per_frames)


def reconstruct_image(data, infowave, pixels_per_line, lines_per_frame=None, reduce=np.sum):
    """Reconstruct a scan or kymograph image from raw data

    Parameters
    ----------
    data : array_like
        Raw data to use for the reconstruction. E.g. photon counts or force samples.
    infowave : array_like
        The famous infowave.
    pixels_per_line : int
        The number of pixels on the fast axis of the scan.
    lines_per_frame : Optional[int]
        The number of pixels on the slow axis of the scan. Only needed for multi-frame scans.
    reduce : callable
        A function which reduces multiple sample into a pixel. Usually `np.sum`
        for photon counts and `np.mean` for force samples.

    Returns
    -------
    np.ndarray
    """
    assert data.size == infowave.size

    # Example infowave:
    #  1 0 0 1 0 1 2 0 1 0 0 1 0 1 0 1 0 0 1 0 2 0 1 0 1 0 0 1 0 1 0 1 2 1 0 0 1
    #              ^ <-----------------------> ^                       ^
    #                       one pixel
    valid_idx = infowave != InfowaveCode.discard
    infowave = infowave[valid_idx]

    # After discard:
    #  1 1 1 2 1 1 1 1 1 2 1 1 1 1 1 2 1 1 1
    #        ^ <-------> ^           ^
    #         pixel_size (i.e. data samples per pixel)
    # This should be:
    #   pixel_sizes = np.diff(np.flatnonzero(infowave == InfowaveCode.pixel_boundary))
    # But for now we assume that every pixel consists of the same number of samples
    pixel_size = np.argmax(infowave) + 1

    def round_up(size, n):
        """Round up `size` to the nearest multiple of `n`"""
        return int(math.ceil(size / n)) * n

    data = data[valid_idx]
    data.resize(round_up(data.size, pixel_size))

    pixels = reduce(data.reshape(-1, pixel_size), axis=1)

    if lines_per_frame is None:
        pixels.resize(round_up(pixels.size, pixels_per_line))
        return pixels.reshape(-1, pixels_per_line)
    else:
        pixels.resize(round_up(pixels.size, pixels_per_line * lines_per_frame))
        return pixels.reshape(-1, lines_per_frame, pixels_per_line).squeeze()


def save_tiff(image, filename, dtype, clip=False):
        """Save an RGB `image` to TIFF

        This is a thin wrapper around `tifffile` with additional safety checks

        Parameters
        ----------
        image : np.array
            Image data. An RGB image is expected to have `shape == (w, h, 3)`.
        filename : str
            The name of the TIFF file where the image will be saved.
        dtype : np.dtype
            The data type of a single color channel in the resulting image.
        clip : bool
            If enabled, the photon count data will be clipped to fit into the desired `dtype`.
            This option is disabled by default: an error will be raise if the data does not fit.
        """
        import tifffile

        info = np.finfo(dtype) if np.dtype(dtype).kind == "f" else np.iinfo(dtype)
        if not clip and (np.min(image) < info.min or np.max(image) > info.max):
            raise RuntimeError(f"Can't safely export image with `dtype={dtype.__name__}` channels."
                               f" Switch to a larger `dtype` in order to safely store everything"
                               f" or pass `force=True` to clip the data.")

        tifffile.imsave(filename, image.astype(dtype))
