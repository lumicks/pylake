import numpy as np
from .mock_json import mock_json
from lumicks.pylake.detail.confocal import ScanMetaData
from lumicks.pylake.detail.image import InfowaveCode
from lumicks.pylake.channel import Continuous, Slice, empty_slice
from lumicks.pylake.kymo import Kymo
from lumicks.pylake.scan import Scan


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


def generate_image_data(image_data, samples_per_pixel, line_padding):
    """Generates the appropriate info_wave and photon_count data for image data.

    Parameters
    ----------
    image_data : array_like
        Image data to generate an infowave for.
    samples_per_pixel : int
        How many samples to divide a pixel over.
    line_padding : int
        Number of "ignore" samples to pad before and after each line.
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
    padding = np.zeros(line_padding, dtype=np.uint8)

    def generate_infowave_line(line):
        """Generate a line of the infowave and pad on both sides with padding"""
        return np.hstack((padding, np.hstack(np.tile(pixelwave, line.shape)), padding))

    def generate_photon_count_line(line):
        """Generate a line of photon counts and pad on both sides with padding"""
        return np.hstack(
            (padding, np.hstack([split_pixel(pixel, samples_per_pixel) for pixel in line]), padding)
        )

    if image_data.ndim == 3:
        # Flattens lines of a multi-frame image. This concatenates the lines of different frames.
        image_data = np.hstack([img for img in image_data])

    return np.hstack([generate_infowave_line(line) for line in image_data.T]), np.hstack(
        [generate_photon_count_line(line) for line in image_data.T]
    )


class MockConfocalFile:
    def __init__(self, infowave, red_channel=None, blue_channel=None, green_channel=None):
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
        start=int(1e9),
        dt=7,
        axes=[0, 1],
        samples_per_pixel=5,
        line_padding=3,
    ):
        """Generate a mock file that can be read by Kymo or Scan"""
        infowave, photon_counts = generate_image_data(image, samples_per_pixel, line_padding)
        json_string = generate_scan_json(
            [
                {
                    "axis": axis,
                    "num of pixels": num_pixels,
                    "pixel size (nm)": pixel_size,
                }
                for pixel_size, axis, num_pixels in zip(pixel_sizes_nm, axes, image.shape[-2:])
            ]
        )

        return (
            MockConfocalFile(
                Slice(Continuous(infowave, start=start, dt=dt)),
                Slice(Continuous(photon_counts, start=start, dt=dt)),
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


def generate_kymo(name, image, pixel_size_nm, start=4, dt=7, samples_per_pixel=5, line_padding=3):
    confocal_file, metadata, stop = MockConfocalFile.from_image(
        image,
        pixel_sizes_nm=[pixel_size_nm],
        axes=[0],
        start=np.int64(start),
        dt=np.int64(dt),
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
    )

    return Kymo(name, confocal_file, start, stop, metadata)


def generate_scan(
    name, scan_data, pixel_sizes_nm, start=4, dt=7, samples_per_pixel=5, line_padding=3
):
    confocal_file, metadata, stop = MockConfocalFile.from_image(
        np.swapaxes(scan_data, -1, -2),
        pixel_sizes_nm=pixel_sizes_nm,
        axes=np.arange(scan_data.ndim),
        start=np.int64(start),
        dt=np.int64(dt),
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
    )

    return Scan(name, confocal_file, start, stop, metadata)
