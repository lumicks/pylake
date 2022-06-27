import enum
import math
import numpy as np
import warnings


class InfowaveCode(enum.IntEnum):
    discard = 0  # this data sample does not contain useful information
    use = 1  # useful data to be used to form a pixel
    pixel_boundary = 2  # useful data and marks the last data sample in a pixel


def discard_zeros(infowave):
    # Example infowave:
    #  1 0 0 1 0 1 2 0 1 0 0 1 0 1 0 1 0 0 1 0 2 0 1 0 1 0 0 1 0 1 0 1 2 1 0 0 1
    #              ^ <-----------------------> ^                       ^
    #                       one pixel
    valid_idx = infowave != InfowaveCode.discard
    subset = infowave[valid_idx]

    # After discard:
    #  1 1 1 2 1 1 1 1 1 2 1 1 1 1 1 2 1 1 1
    #        ^ <-------> ^           ^
    #         pixel_size (i.e. data samples per pixel)
    return subset, valid_idx


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


def line_timestamps_image(time_stamps, infowave, pixels_per_line):
    """Determine the starting timestamps for lines in the kymograph

    Parameters
    ----------
    time_stamps : array_like
        Timestamps corresponding to the infowave.
    infowave : array_like
        The infamous infowave.
    pixels_per_line : int
        The number of pixels on the fast axis of the scan.
    """
    infowave, valid_idx = discard_zeros(infowave)
    time_stamps = time_stamps[valid_idx]
    pixel_start_idx = np.flatnonzero(infowave == InfowaveCode.pixel_boundary)
    pixel_start_idx = np.concatenate(([0], pixel_start_idx + 1))
    return time_stamps[pixel_start_idx[0:-1:pixels_per_line]]


def seek_timestamp_next_line(infowave):
    """Seeks the timestamp beyond the first line. Used for repairing kymos with truncated start."""
    time = infowave.timestamps
    infowave = infowave.data

    # Discard unused acquisition
    infowave, valid_idx = discard_zeros(infowave)
    time = time[valid_idx]

    # Fetch start of pixels
    (pixel_ends,) = np.where(infowave == InfowaveCode.pixel_boundary)
    pixel_start = time[pixel_ends[:-1] + 1]

    # Determine pixel times
    dts = np.diff(pixel_start)

    # Long ones indicate a line. Find the first one.
    mx = np.max(dts)
    mn = np.min(dts)
    pixel_time_threshold = (mx + mn) / 2
    idx = np.argmax(dts > pixel_time_threshold)

    return pixel_start[idx + 1]


def round_up(size, n):
    """Round up `size` to the nearest multiple of `n`"""
    return int(math.ceil(size / n)) * n


def round_down(size, n):
    """Round down `size` to the nearest multiple of `n`"""
    return (size // n) * n


def reshape_reconstructed_image(pixels, shape):
    """Reshape reconstructed image data from 1D array into appropriate shape for plotting

    Parameters
    __________
    pixels : array_like
        Image data
    shape : array_like
        The shape of the image ([optional: pixels on slow axis], pixels on fast axis)
    """
    resized_pixels = np.zeros(round_up(pixels.size, np.prod(shape)), dtype=pixels.dtype)
    resized_pixels[: pixels.size] = pixels
    return resized_pixels.reshape(-1, *shape)


def reconstruct_image_sum(data, infowave, shape):
    """Rapidly reconstruct a scan or kymograph image from raw data summing pixels where the infowave equates to 1.
    See reconstruct_image for more information.
    *NOTE*: this function should not be used to reconstruct timestamps as the result will overflow.

    Parameters
    ----------
    data : array_like
        Raw data to use for the reconstruction. E.g. photon counts or force samples.
    infowave : array_like
        The infamous infowave.
    shape : array_like
        The shape of the image ([optional: pixels on slow axis], pixels on fast axis)

    Returns
    -------
    np.ndarray
    """
    assert data.size == infowave.size
    subset, valid_idx = discard_zeros(infowave)
    cumulative = np.cumsum(data[valid_idx])
    pixel_ends = cumulative[subset == InfowaveCode.pixel_boundary]
    pixels = np.hstack((pixel_ends[0], np.diff(pixel_ends)))
    return reshape_reconstructed_image(pixels, shape)


def reconstruct_image(data, infowave, shape, reduce=np.sum):
    """Reconstruct a scan or kymograph image from raw data

    Parameters
    ----------
    data : array_like
        Raw data to use for the reconstruction. E.g. photon counts or force samples.
    infowave : array_like
        The infamous infowave.
    shape: array_like
        The shape of the image ([optional: pixels on slow axis], pixels on fast axis)
    reduce : callable
        A function which reduces multiple sample into a pixel. Usually `np.sum`
        for photon counts and `np.mean` for force samples.

    Returns
    -------
    np.ndarray
    """
    assert data.size == infowave.size
    subset, valid_idx = discard_zeros(infowave)
    # This should be:
    #   pixel_sizes = np.diff(np.flatnonzero(infowave == InfowaveCode.pixel_boundary))
    # But for now we assume that every pixel consists of the same number of samples
    pixel_size = np.argmax(subset) + 1
    resized_data = data[valid_idx][: round_down(subset.size, pixel_size)]
    pixels = reduce(resized_data.reshape(-1, pixel_size), axis=1)
    return reshape_reconstructed_image(pixels, shape)


def save_tiff(
    image, filename, dtype, clip=False, pixel_sizes_um=(1e-3, 1e-3), pixel_time_seconds=1.0
):
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
    pixel_sizes_um : array_like
        Pixel sizes in [um]
    pixel_time_seconds : float
        Pixel acquisition time [s]
    """
    import tifffile

    info = np.finfo(dtype) if np.dtype(dtype).kind == "f" else np.iinfo(dtype)
    if not clip and (np.min(image) < info.min or np.max(image) > info.max):
        raise RuntimeError(
            f"Can't safely export image with `dtype={dtype.__name__}` channels."
            f" Switch to a larger `dtype` in order to safely store everything"
            f" or pass `force=True` to clip the data."
        )

    pixel_size_x, pixel_size_y = (
        pixel_sizes_um[0],
        pixel_sizes_um[1] if len(pixel_sizes_um) == 2 else pixel_sizes_um[0],
    )

    tiff_kwargs = {
        "metadata": {"PixelTime": pixel_time_seconds, "PixelTimeUnit": "s"},
        "photometric": "rgb",
    }
    if pixel_size_x:
        tiff_kwargs["resolution"] = (1e4 / pixel_size_x, 1e4 / pixel_size_y, "CENTIMETER")

    tifffile.imwrite(filename, image.astype(dtype), **tiff_kwargs)


def histogram_rows(image, pixels_per_bin, pixel_width):
    bin_width = pixels_per_bin * pixel_width  # in physical units
    n_rows = image.shape[0]
    if pixels_per_bin > n_rows:
        raise ValueError("bin size is larger than the available pixels")

    n_bins = n_rows // pixels_per_bin
    remainder = n_rows % pixels_per_bin
    if remainder != 0:
        warnings.warn(
            f"{n_rows} pixels is not divisible by {pixels_per_bin}, final bin only contains {remainder} pixels"
        )
        pad = np.zeros((pixels_per_bin - remainder, image.shape[1]))
        image = np.vstack((image, pad))
        n_bins += 1

    counts = image.reshape((n_bins, -1)).sum(axis=1)
    edges = np.arange(n_bins) * bin_width
    widths = np.diff(np.hstack((edges, n_rows * pixel_width)))
    return edges, counts, widths


def make_image_title(image_object, frame, show_name=True):
    name = f"{image_object.name} " if show_name else ""
    return (
        name
        if image_object.num_frames == 1
        else f"{name}[frame {frame + 1} / {image_object.num_frames}]"
    )
