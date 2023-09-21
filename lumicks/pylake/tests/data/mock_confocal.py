from typing import List
from numbers import Integral
from dataclasses import dataclass

import numpy as np

from lumicks.pylake.kymo import Kymo
from lumicks.pylake.scan import Scan
from lumicks.pylake.channel import Slice, Continuous, empty_slice
from lumicks.pylake.detail.image import InfowaveCode
from lumicks.pylake.detail.confocal import ScanMetaData, ConfocalImage

from .mock_json import mock_json


def generate_scan_json(axes):
    """Generate a mock JSON for a Scan or Kymo.

    Parameters
    ----------
    axes : List[Dict]
        List of dictionaries with an element for each axis. These dictionaries need the following
        fields:
        "axis" : int
            Axis order.
        "num of pixels" : int
            Number of pixels along this axis.
        "pixel size (nm)" : float
            Pixel size along this axis.
    """
    axes_metadata = [
        {
            "axis": int(axis["axis"]),
            "cereal_class_version": 1,
            "num of pixels": axis["num of pixels"],
            "pixel size (nm)": axis["pixel size (nm)"],
            "scan time (ms)": 0,
            "scan width (um)": axis["pixel size (nm)"] * axis["num of pixels"] / 1000.0 + 0.5,
        }
        for axis in axes
    ]

    return mock_json(
        {
            "cereal_class_version": 1,
            "fluorescence": True,
            "force": False,
            "scan count": 0,
            "scan volume": {
                "center point (um)": {"x": 58.075877109272604, "y": 31.978375270573267, "z": 0},
                "cereal_class_version": 1,
                "pixel time (ms)": 0.2,
                "scan axes": axes_metadata,
            },
        },
    )


def generate_image_data(image_data, samples_per_pixel, line_padding, multi_color=False):
    """Generates the appropriate info_wave and photon_count data for image data.

    Parameters
    ----------
    image_data : array_like
        Image data to generate an infowave for.
    samples_per_pixel : int
        How many samples to divide a pixel over.
    line_padding : int
        Number of "ignore" samples to pad before and after each line.
    multi_color : bool
        Does the `image_data` consist of a multi color channel image? If so,
        return individual photon counts for every color channel.
    """

    def split_pixel(x, num_samples):
        """Splits a pixel into separate samples"""
        result = np.zeros((num_samples,), dtype=np.uint32)
        for i in range(num_samples - 1):
            if x > 0:
                result[i] = np.random.randint(x)
                x -= result[i]
            else:
                result[i] = 0
        result[num_samples - 1] = x
        np.random.shuffle(result)
        return result

    pixelwave = np.ones(samples_per_pixel, dtype=np.uint8) * InfowaveCode.use
    pixelwave[-1] = InfowaveCode.pixel_boundary
    padding = np.ones(line_padding, dtype=np.uint8) * InfowaveCode.discard

    def generate_infowave_line(number_of_pixels):
        """Generate a line of the infowave and pad on both sides with padding"""
        return np.hstack((padding, np.hstack(np.tile(pixelwave, number_of_pixels)), padding))

    def generate_photon_count_line(line):
        """Generate a line of photon counts and pad on both sides with padding"""
        return np.hstack(
            (padding, np.hstack([split_pixel(pixel, samples_per_pixel) for pixel in line]), padding)
        )

    # Flatten lines of a multi-frame image. Images with multiple frames and one color have 3 and
    # with multiple colors 4 dimensions, respectively.
    if image_data.ndim == 3 + multi_color:
        # Concatenate the lines of different frames.
        image_data = np.hstack([img for img in image_data])

    # Ensure to have a third dimension with the color channels, as we will iterate over them
    image_data = np.expand_dims(image_data, -1) if not multi_color else image_data

    # Create an infowave and photon counts for all color channels based on the image_data and return
    # a tuple consisting of (infowave, (*photon_counts))
    pixels_per_line, number_of_lines = image_data.shape[:2]
    return (
        np.tile(generate_infowave_line(pixels_per_line), number_of_lines),
        tuple(
            np.hstack([generate_photon_count_line(line) for line in channel.T])
            for channel in np.moveaxis(image_data, -1, 0)
        ),
    )


class MockConfocalFile:
    def __init__(self, infowave, red_channel=None, green_channel=None, blue_channel=None):
        self.infowave = infowave
        self.red_photon_count = red_channel if red_channel is not None else empty_slice
        self.green_photon_count = green_channel if green_channel is not None else empty_slice
        self.blue_photon_count = blue_channel if blue_channel is not None else empty_slice

    def __getitem__(self, key):
        if key == "Info wave":
            return {"Info wave": self.infowave}

    @staticmethod
    def from_image(
        image,
        pixel_sizes_nm,
        axes=None,
        start=int(20e9),
        dt=int(62.5e6),
        samples_per_pixel=5,
        line_padding=3,
        multi_color=False,
    ):
        """Generate a mock file that can be read by Kymo or Scan"""
        axes = [0, 1] if axes is None else axes

        if len(axes) == 2 and axes[0] < axes[1]:
            # The fast axis of the physical scannning process has the lower physical axis number
            # (e.g. x) and comes before the slow axis with the greater physical axis number (e.g. y
            # or z). The convention of indexing numpy arrays is to have the slow indexed y axis
            # before the fast indexed x axis. Therefore, flip the axes to ensure correct indexing
            # order when creating the infowave:
            image = image.swapaxes(-1 - multi_color, -2 - multi_color)
        infowave, photon_counts = generate_image_data(
            image, samples_per_pixel, line_padding, multi_color=multi_color
        )
        json_string = generate_scan_json(
            [
                {
                    "axis": axis,
                    "num of pixels": num_pixels,
                    "pixel size (nm)": pixel_size,
                }
                for pixel_size, axis, num_pixels in zip(
                    pixel_sizes_nm, axes, image.shape[-2 - multi_color : image.ndim - multi_color]
                )
            ]
        )

        return (
            MockConfocalFile(
                Slice(Continuous(infowave, start=start, dt=dt)),
                *(Slice(Continuous(channel, start=start, dt=dt)) for channel in photon_counts),
            ),
            ScanMetaData.from_json(json_string),
            start + len(infowave) * dt,
        )

    @staticmethod
    def from_streams(
        start,
        dt,
        axes,
        num_pixels,
        pixel_sizes_nm,
        infowave,
        red_photon_counts=None,
        blue_photon_counts=None,
        green_photon_counts=None,
    ):
        def make_slice(data):
            if data is None:
                return empty_slice
            else:
                return Slice(Continuous(data, start, dt))

        if axes == [] and num_pixels == [] and pixel_sizes_nm == []:
            json_string = generate_scan_json([])
        else:
            json_string = generate_scan_json(
                [
                    {
                        "axis": axis,
                        "num of pixels": num_pixels,
                        "pixel size (nm)": pixel_size,
                    }
                    for (axis, pixel_size, num_pixels) in zip(axes, pixel_sizes_nm, num_pixels)
                ]
            )

        return (
            MockConfocalFile(
                infowave=make_slice(infowave),
                red_channel=make_slice(red_photon_counts),
                blue_channel=make_slice(blue_photon_counts),
                green_channel=make_slice(green_photon_counts),
            ),
            ScanMetaData.from_json(json_string),
            start + len(infowave) * dt,
        )


def generate_kymo(
    name,
    image,
    pixel_size_nm=10.0,
    start=int(20e9),
    dt=int(62.5e6),
    samples_per_pixel=5,
    line_padding=3,
):
    """Generate a kymo based on provided image data"""
    return _generate_confocal(
        name,
        Kymo,
        image,
        multi_color=image.ndim > 2,
        pixel_sizes_nm=[pixel_size_nm],
        axes=[0],
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
    )


def _scan_axes(image, axes):
    """Provide a central definition of default axes argument for scans"""
    return np.arange(min(2, image.ndim)) if axes is None else axes


def generate_scan(
    name,
    image,
    pixel_sizes_nm,
    axes=None,
    start=int(20e9),
    dt=int(62.5e6),
    samples_per_pixel=5,
    line_padding=3,
    multi_color=False,
):
    """Generate a scan based on provided image data"""
    return _generate_confocal(
        name,
        Scan,
        image,
        multi_color=multi_color,
        pixel_sizes_nm=pixel_sizes_nm,
        axes=_scan_axes(image, axes),
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
    )


def _generate_confocal(
    name,
    confocal_class,
    image,
    multi_color,
    pixel_sizes_nm,
    axes,
    start,
    dt,
    samples_per_pixel,
    line_padding,
):
    """Generate a kymo or a scan based on provided image data"""
    start = np.int64(start)
    dt = np.int64(dt)

    confocal_file, metadata, stop = MockConfocalFile.from_image(
        image,
        pixel_sizes_nm=pixel_sizes_nm,
        axes=axes,
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
        multi_color=multi_color,
    )

    return confocal_class(name, confocal_file, start, stop, metadata)


def generate_kymo_with_ref(
    name,
    image,
    pixel_size_nm=10.0,
    start=int(20e9),
    dt=int(62.5e6),
    samples_per_pixel=5,
    line_padding=3,
):
    """Generate a kymo and a corresponding reference"""
    kymo = generate_kymo(name, image, pixel_size_nm, start, dt, samples_per_pixel, line_padding)
    ref = _generate_confocal_reference(
        Kymo,
        kymo.file,
        kymo._metadata,
        image,
        multi_color=image.ndim > 2,
        axes=[0],
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
    )
    return kymo, ref


def generate_scan_with_ref(
    name,
    image,
    pixel_sizes_nm,
    axes=None,
    start=int(20e9),
    dt=int(62.5e6),
    samples_per_pixel=5,
    line_padding=3,
    multi_color=False,
):
    """Generate a scan and a corresponding reference"""
    scan = generate_scan(
        name, image, pixel_sizes_nm, axes, start, dt, samples_per_pixel, line_padding, multi_color
    )
    ref = _generate_confocal_reference(
        Scan,
        scan.file,
        scan._metadata,
        image,
        multi_color=multi_color,
        axes=_scan_axes(image, axes),
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
    )
    return scan, ref


def _generate_confocal_reference(
    confocal_class,
    confocal_file,
    metadata,
    image,
    multi_color,
    axes,
    start,
    dt,
    samples_per_pixel,
    line_padding,
):
    """Generate a reference for a kymo or a scan"""
    start = np.int64(start)
    dt = np.int64(dt)

    # Assume the x axis to be the slow and the y axis to be the fast indexed one: The x axis
    # contains the `lines_per_frame` and the y axis the `pixels_per_line`:
    number_of_frames, pixels_per_line, lines_per_frame = (
        [1, *image.shape[:2]] if len(image.shape) == 2 + multi_color else image.shape[:3]
    )

    first_physical_axis_fast = len(axes) == 2 and axes[0] < axes[1]
    if first_physical_axis_fast:
        # The x axis contains the `pixels_per_line` and the y axis the `lines_per_frame`. Therefore,
        # switch the values of the variables.
        lines_per_frame, pixels_per_line = pixels_per_line, lines_per_frame

    timestamps, timestamp_ranges, timestamp_ranges_deadtime = generate_timestamps(
        number_of_frames=number_of_frames,
        lines_per_frame=lines_per_frame,
        pixels_per_line=pixels_per_line,
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
        scan=confocal_class == Scan,
        x_axis_fast=first_physical_axis_fast,
    )

    times_ref = TimeStampsRef(
        data=timestamps,
        timestamp_ranges=timestamp_ranges,
        timestamp_ranges_deadtime=timestamp_ranges_deadtime,
        dt=dt,
        pixel_time_seconds=dt * samples_per_pixel * 1e-9,
        line_time_seconds=dt * (pixels_per_line * samples_per_pixel + line_padding * 2) * 1e-9,
    )

    infowave_ref = InfoWaveRef(
        data=confocal_file.infowave,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
    )

    metadata_ref = MetaDataRef(
        number_of_frames=number_of_frames,
        lines_per_frame=lines_per_frame,
        pixels_per_line=pixels_per_line,
        fast_axis={0: "X", 1: "Y", 2: "Z"}[axes[0]],
        num_pixels=[axes.num_pixels for axes in metadata.ordered_axes],
        pixelsize_um=[axes.pixel_size_um for axes in metadata.ordered_axes],
        center_point_um=metadata.center_point_um,
    )

    ref = ConfocalRef(
        confocal_class=confocal_class,
        image=image if multi_color else np.repeat(image, 3).reshape(*image.shape, 3),
        timestamps=times_ref,
        infowave=infowave_ref,
        metadata=metadata_ref,
        start=start,
        stop=start + len(confocal_file.infowave) * dt,
    )

    return ref


def generate_timestamps(
    number_of_frames,
    lines_per_frame,
    pixels_per_line,
    start,
    dt,
    samples_per_pixel,
    line_padding,
    scan=False,
    x_axis_fast=True,
):
    """Calculate reference timestamps of a kymo or scan created with `MockConfocal.from_image()`

    Parameters
    ----------
    number_of_frames : int
        The number of frames to create timestamps for.
    lines_per_frame : int
        The number of lines per frame to create timestamps for.
    pixels_per_line : int
        The number of pixels per line to create timestamps for.
    start : int
        The start time.
    dt : int
        The increment for each timestamp.
    samples_per_pixel : int
        The number of timestamps to be used to calculate the mean timestamp for one pixel.
    line_padding : int
        The number of (finally unused) timestamps added to each end of one line of timestamps.
    scan : bool
        Create frame timestamp ranges instead of line timestamp ranges. Defaults to False. If
        `number_of_frames` > 1 this will default to True.
    x_axis_fast : bool
        The x axis of the returned timestamps array is the fast indexed one (default). If False, the
        y axis is the fast indexed and the x axis the slow indexed one.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        The pixel timestamps and the line (kymo) or frame (scan) timestamp ranges.
    """
    if not isinstance(start, Integral):
        raise TypeError("`start` needs to be an integer")
    if not isinstance(dt, Integral):
        raise TypeError("`dt` needs to be an integer")
    if start < 0:
        raise ValueError("`start` needs to be non negative")
    if dt <= 0:
        raise ValueError("`dt` needs to be positive")

    # Create timestamps
    timestamps_per_line = pixels_per_line * samples_per_pixel + line_padding * 2
    number_of_timestamps = number_of_frames * lines_per_frame * timestamps_per_line
    if start + int(number_of_timestamps + 1) * dt > 2**63:
        raise OverflowError("timestamps are too big for int64")
    timestamps_padded = np.arange(
        start, start + number_of_timestamps * dt - dt // 2, dt, dtype=np.int64
    ).reshape((number_of_frames, lines_per_frame, timestamps_per_line))

    # Remove line padding timestamps and get the line or frame start/stop timestamps
    timestamps = timestamps_padded[
        :, :, slice(line_padding, -line_padding) if line_padding else slice(None)
    ]

    if scan or number_of_frames > 1:
        # frame timestamp ranges
        timestamp_ranges = np.c_[timestamps[:, 0, 0], timestamps[:, -1, -1] + dt]
        timestamp_ranges_deadtime = np.c_[
            timestamps_padded[:, 0, 0], timestamps_padded[:, -1, -1] + dt
        ]
    else:
        # line timestamp ranges
        timestamp_ranges = timestamps[:, :, [0, -1]].squeeze()
        timestamp_ranges[:, 1] += dt
        # we pad left and right, but we define the deadtime-included time as the time of the
        # first true sample, up to the next true sample (hence the shift by the left padding)
        timestamp_ranges_deadtime = timestamps_padded[:, :, [0, -1]].squeeze()
        timestamp_ranges_deadtime[:, 1] += dt
    timestamp_ranges_deadtime += dt * line_padding

    # Get pixel timestamps by calculating mean of all pixels (i.e. mean of first and last sample)
    timestamps = timestamps.reshape(
        number_of_frames * lines_per_frame * pixels_per_line, samples_per_pixel
    )
    timestamps = (
        (np.uint64(timestamps[:, 0]) + np.uint64(timestamps[:, -1])) // np.uint64(2)
    ).astype(np.int64)
    timestamps = timestamps.reshape(number_of_frames, lines_per_frame, pixels_per_line)
    timestamps = timestamps if x_axis_fast else timestamps.swapaxes(2, 1)

    return timestamps.squeeze(), timestamp_ranges, timestamp_ranges_deadtime


@dataclass(frozen=True)
class MetaDataRef:
    """
    Notes
    -----
    For Kymos the number of lines can be accessed via `lines_per_frame`.
    """

    number_of_frames: int
    lines_per_frame: int
    pixels_per_line: int
    fast_axis: str
    num_pixels: List[float]
    pixelsize_um: List[float]
    center_point_um: List[float]


@dataclass(frozen=True)
class TimeStampsRef:
    """
    Notes
    -----
    For Kymos `timestamp_ranges` contain frame timestamp ranges.
    For Scans `timestamp_ranges` contain line timestamp ranges.
    """

    data: np.ndarray
    timestamp_ranges: np.ndarray
    timestamp_ranges_deadtime: np.ndarray
    dt: int
    pixel_time_seconds: float
    line_time_seconds: float


@dataclass(frozen=True)
class InfoWaveRef:
    data: np.ndarray
    samples_per_pixel: int
    line_padding: int


@dataclass(frozen=True)
class ConfocalRef:
    confocal_class: ConfocalImage
    image: np.ndarray
    timestamps: TimeStampsRef
    infowave: InfoWaveRef
    metadata: MetaDataRef
    start: int
    stop: int
