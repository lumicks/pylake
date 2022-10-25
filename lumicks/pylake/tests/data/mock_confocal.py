from dataclasses import dataclass

import numpy as np
from lumicks.pylake.channel import Continuous, Slice, empty_slice
from lumicks.pylake.detail.confocal import ScanMetaData
from lumicks.pylake.detail.image import InfowaveCode
from lumicks.pylake.kymo import Kymo
from lumicks.pylake.scan import Scan

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

    if image_data.ndim == 3 + multi_color:
        # Flattens lines of a multi-frame image. This concatenates the lines of different frames.
        image_data = np.hstack([img for img in image_data])
    image_data = np.expand_dims(image_data, -1) if not multi_color else image_data

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
        axes=[0, 1],
        start=int(1e9),
        dt=7,
        samples_per_pixel=5,
        line_padding=3,
        multi_color=False,
    ):
        """Generate a mock file that can be read by Kymo or Scan"""
        if len(axes) == 2 and axes[0] < axes[1]:
            # First (and fast) axis of the scannning has a lower pyhsical axis number than the
            # second (and slow) axis (xy, xz, or yz). The reconstructed image always has the slow
            # axis as first and the fast axis as second. Therefore, flip the axes:
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
    start=20e9,
    dt=62.5e9,
    samples_per_pixel=5,
    line_padding=3,
    with_ref=False,
):
    return generate_confocal(
        name,
        image,
        scan=False,
        pixel_sizes_nm=[pixel_size_nm],
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
        with_ref=with_ref,
        multi_color=image.ndim > 2,
    )


def generate_scan(
    name,
    image,
    pixel_sizes_nm,
    axes=None,
    start=20e9,
    dt=62.5e9,
    samples_per_pixel=5,
    line_padding=3,
    with_ref=False,
    multi_color=False,
):
    return generate_confocal(
        name,
        image,
        scan=True,
        pixel_sizes_nm=pixel_sizes_nm,
        axes=axes,
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
        with_ref=with_ref,
        multi_color=multi_color,
    )


def generate_confocal(
    name,
    image,
    scan=False,
    pixel_sizes_nm=[10.0],
    axes=None,
    start=20e9,
    dt=62.5e9,
    samples_per_pixel=5,
    line_padding=3,
    with_ref=False,
    multi_color=False,
):
    start = np.int64(start)
    dt = np.int64(dt)
    axes = (np.arange(min(2, image.ndim)) if scan else [0]) if axes is None else axes

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

    confocal_class = Scan if scan else Kymo
    confocal = confocal_class(name, confocal_file, start, stop, metadata)

    if not with_ref:
        return confocal

    def fast_first():
        return len(axes) == 2 and axes[0] < axes[1]

    image_shape = (image.shape[:-1] if multi_color else image.shape) if scan else image.shape[:2]
    if fast_first():
        image_shape = (
            image_shape[::-1] if len(image_shape) == 2 else (image_shape[0], *image_shape[:-3:-1])
        )
    timestamp_ranges, timestamps = reference_timestamps(
        image_shape,
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
        scan=scan,
    )
    if fast_first():
        timestamps = timestamps.swapaxes(-1, -2)

    ref_class = ScanRef if scan else KymoRef
    ref = ref_class(
        image=image,
        start=start,
        stop=stop,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
        metadata=metadata,
        infowave=confocal_file.infowave,
        timestamps=timestamps,
        timestamp_ranges=timestamp_ranges,
        multi_color=multi_color,
    )

    return confocal, ref


def reference_timestamps(channel_shape, start, dt, samples_per_pixel, line_padding, scan=False):
    """Calculate reference timestamps of a kymo or scan created with `MockConfocal.from_image()`

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        The line (kymo) or frame (scan) timestamp ranges and the pixel timestamps.
    """
    if len(channel_shape) == 2:
        number_of_frames = 1
        pixels_per_line, lines_per_frame = channel_shape
    else:
        number_of_frames, pixels_per_line, lines_per_frame = channel_shape

    # Create timestamps
    timestamps_per_line = pixels_per_line * samples_per_pixel + line_padding * 2
    number_of_timestamps = number_of_frames * lines_per_frame * timestamps_per_line
    timestamps = np.arange(start, start + number_of_timestamps * dt - dt / 2, dt).astype(np.int64)

    # Remove line padding timestamps and get the line or frame start/stop timestamps
    timestamps = timestamps.reshape((number_of_frames, lines_per_frame, timestamps_per_line))[
        :, :, slice(line_padding, -line_padding) if line_padding else slice(None)
    ]

    if scan or len(channel_shape) == 3:
        # frame timestamp ranges
        timestamp_ranges = np.c_[timestamps[:, 0, 0], timestamps[:, -1, -1] + dt].astype(np.int64)
    else:
        # line timestamp ranges
        timestamp_ranges = timestamps[:, :, [0, -1]].squeeze()
        timestamp_ranges[:, 1] += dt

    # Get pixel timestamps by calculating mean of all pixels (i.e. mean of first and last sample)
    timestamps = timestamps.reshape(
        number_of_frames * lines_per_frame * pixels_per_line, samples_per_pixel
    )
    timestamps = ((timestamps[:, 0] + timestamps[:, -1]) / 2).astype(np.int64)
    timestamps = timestamps.reshape(number_of_frames, lines_per_frame, pixels_per_line).swapaxes(
        2, 1
    )

    return timestamp_ranges, timestamps.squeeze()


@dataclass(frozen=True)
class ConfocalRef:
    image: np.ndarray
    start: int
    stop: int
    dt: int
    samples_per_pixel: int
    line_padding: int
    metadata: ScanMetaData
    infowave: np.ndarray
    timestamps: np.ndarray
    timestamp_ranges: np.ndarray
    multi_color: bool

    @property
    def center_point_um(self):
        return self.metadata.center_point_um

    @property
    def fast_axis(self):
        return self.metadata.scan_axes[0].axis_label

    @property
    def shape(self):
        return self.image.shape

    @property
    def line_time_seconds(self):
        return (self.timestamp_ranges[1, 0] - self.timestamp_ranges[0, 0]) * 1e-9

    @property
    def pixel_time_seconds(self):
        return (self.timestamps[1, 0] - self.timestamps[0, 0]) * 1e-9

    @property
    def line_start_timestamps(self):
        return self.timestamp_ranges[:, 0]

    def get_line_idx(self, time_s):
        """Return the index of the next line starting >= `time_ns`. If there is no next line, return
        the number of total lines.
        """
        time_ns = time_s * 1e9
        time_ns += self.start if time_ns >= 0 else self.stop
        return np.searchsorted(self.line_start_timestamps, time_ns, side="left")

    def get_timestamp_ranges(self, include_dead_time=False):
        if include_dead_time:
            timestamp_ranges = self.timestamp_ranges[:, [0, 0]]
            timestamp_ranges[:, 1] += (self.line_time_seconds * 1e9).astype(np.int64)
            return timestamp_ranges
        else:
            return self.timestamp_ranges


@dataclass(frozen=True)
class KymoRef(ConfocalRef):
    @property
    def pixels_per_line(self):
        return self.image.shape[0]

    @property
    def number_of_lines(self):
        return self.image.shape[1]

    @property
    def pixelsize_um(self):
        return self.metadata.scan_axes[0].pixel_size_um

    @property
    def size_um(self):
        return self.pixelsize_um * self.pixels_per_line

    @property
    def line_timestamp_ranges(self):
        return self.timestamp_ranges

    def get_line_timestamp_ranges(self, include_dead_time=False):
        return self.get_timestamp_ranges(include_dead_time=include_dead_time)


@dataclass(frozen=True)
class ScanRef(ConfocalRef):
    @property
    def multi_frame(self):
        return self.image.ndim - self.multi_color == 3

    @property
    def number_of_frames(self):
        return self.image.shape[0] if self.multi_frame else 1

    @property
    def lines_per_frame(self):
        return self.image.shape[self.metadata.scan_order[0] + self.multi_frame]

    @property
    def pixels_per_line(self):
        return self.image.shape[self.metadata.scan_order[1] + self.multi_frame]

    @property
    def pixelsize_um(self):
        return [axes.pixel_size_um for axes in self.metadata.ordered_axes]

    @property
    def num_pixels(self):
        return [axes.num_pixels for axes in self.metadata.ordered_axes]

    @property
    def size_um(self):
        return [size * num for size, num in zip(self.pixelsize_um, self.num_pixels)]

    @property
    def line_timestamp_ranges(self):
        return self.timestamp_ranges

    def get_frame_timestamp_ranges(self, include_dead_time=False):
        return self.get_timestamp_ranges(include_dead_time=include_dead_time)
