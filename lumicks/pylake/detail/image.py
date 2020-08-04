import enum
import math
import numpy as np


class ImageMetadata:
    """Image metadata

    Parameters
    ----------
    pixel_size_x : float
        Horizontal pixel size [nm]
    pixel_size_y : float
        Vertical pixel size [nm]
        When omitted, assumed to be identical to pixel_size_x
    pixel_time : float
        How long does pixel acquisition take [ms]
    """""
    def __init__(self, pixel_size_x=1.0, pixel_size_y=None, pixel_time=1.0):
        self._pixel_time = pixel_time
        self._pixel_size_x = pixel_size_x
        if pixel_size_y:
            self._pixel_size_y = pixel_size_y
        else:
            self._pixel_size_y = pixel_size_x

    @staticmethod
    def from_dataset(json):
        """
        Fetch metadata from json structure

        Parameters
        ----------
        json : dict
            json structure containing kymograph metadata
        """
        if json:
            pixel_size = json["scan volume"]["scan axes"][0]["pixel size (nm)"]
            pixel_time = json["scan volume"]["pixel time (ms)"]
            return ImageMetadata(pixel_size_x=pixel_size, pixel_time=pixel_time)
        else:
            return ImageMetadata()

    @property
    def resolution(self):
        """X, Y resolution in pixels per cm followed by unit specification accepted by Tifffile"""
        # TIFF only supports centimeters and inches as valid units, hence we convert from nm => cm
        return 1e7 / self._pixel_size_x, 1e7 / self._pixel_size_y, "CENTIMETER"

    @property
    def metadata(self):
        """Dictionary with metadata"""
        pixel_time = self._pixel_time * 1e-3  # ms => s
        return {'PixelTime': pixel_time, 'PixelTimeUnit': 's'}


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
    pixel_start_idx = np.flatnonzero(infowave == InfowaveCode.pixel_boundary)
    pixel_start_idx = np.concatenate(([0], pixel_start_idx + 1))

    return time_stamps[pixel_start_idx[0:-1:pixels_per_line]]


def seek_timestamp_next_line(infowave):
    """Seeks the timestamp beyond the first line. Used for repairing kymos with truncated start."""
    time = infowave.timestamps
    infowave = infowave.data

    # Discard unused acquisition
    mask = infowave != InfowaveCode.discard
    infowave = infowave[mask]
    time = time[mask]

    # Fetch start of pixels
    pixel_ends, = np.where(infowave == InfowaveCode.pixel_boundary)
    pixel_start = time[pixel_ends[:-1] + 1]

    # Determine pixel times
    dts = np.diff(pixel_start)

    # Long ones indicate a line. Find the first one.
    mx = np.max(dts)
    mn = np.min(dts)
    pixel_time_threshold = (mx + mn) / 2
    idx = np.argmax(dts > pixel_time_threshold)

    return pixel_start[idx + 1]


def reconstruct_image_sum(data, infowave, pixels_per_line, lines_per_frame=None):
    """Rapidly reconstruct a scan or kymograph image from raw data summing pixels where the infowave equates to 1.
    See reconstruct_image for more information.

    Parameters
    ----------
    data : array_like
        Raw data to use for the reconstruction. E.g. photon counts or force samples.
    infowave : array_like
        The infamous infowave.
    pixels_per_line : int
        The number of pixels on the fast axis of the scan.
    lines_per_frame : Optional[int]
        The number of pixels on the slow axis of the scan. Only needed for multi-frame scans.

    Returns
    -------
    np.ndarray
    """
    assert data.size == infowave.size

    mask = infowave != InfowaveCode.discard
    cumulative = np.cumsum(data[mask])
    subset = infowave[mask]
    pixel_ends = cumulative[subset == InfowaveCode.pixel_boundary]
    pixels = np.hstack((pixel_ends[0], np.diff(pixel_ends)))

    def round_up(size, n):
        """Round up `size` to the nearest multiple of `n`"""
        return int(math.ceil(size / n)) * n

    if lines_per_frame is None:
        resized_pixels = np.zeros(round_up(pixels.size, pixels_per_line))
        resized_pixels[:pixels.size] = pixels
        return resized_pixels.reshape(-1, pixels_per_line)
    else:
        resized_pixels = np.zeros(round_up(pixels.size, pixels_per_line * lines_per_frame))
        resized_pixels[:pixels.size] = pixels
        return resized_pixels.reshape(-1, lines_per_frame, pixels_per_line).squeeze()


def reconstruct_image(data, infowave, pixels_per_line, lines_per_frame=None, reduce=np.sum):
    """Reconstruct a scan or kymograph image from raw data

    Parameters
    ----------
    data : array_like
        Raw data to use for the reconstruction. E.g. photon counts or force samples.
    infowave : array_like
        The infamous infowave.
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

    resized_data = np.zeros(round_up(infowave.size, pixel_size))
    resized_data[:infowave.size] = data[valid_idx]
    pixels = reduce(resized_data.reshape(-1, pixel_size), axis=1)

    if lines_per_frame is None:
        resized_pixels = np.zeros(round_up(pixels.size, pixels_per_line))
        resized_pixels[:pixels.size] = pixels
        return resized_pixels.reshape(-1, pixels_per_line)
    else:
        resized_pixels = np.zeros(round_up(pixels.size, pixels_per_line * lines_per_frame))
        resized_pixels[:pixels.size] = pixels
        return resized_pixels.reshape(-1, lines_per_frame, pixels_per_line).squeeze()


def save_tiff(image, filename, dtype, clip=False, metadata=ImageMetadata()):
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
    metadata : ImageMetadata
    """
    import tifffile

    info = np.finfo(dtype) if np.dtype(dtype).kind == "f" else np.iinfo(dtype)
    if not clip and (np.min(image) < info.min or np.max(image) > info.max):
        raise RuntimeError(f"Can't safely export image with `dtype={dtype.__name__}` channels."
                           f" Switch to a larger `dtype` in order to safely store everything"
                           f" or pass `force=True` to clip the data.")

    tifffile.imsave(filename, image.astype(dtype), resolution=metadata.resolution,
                    metadata=metadata.metadata)

