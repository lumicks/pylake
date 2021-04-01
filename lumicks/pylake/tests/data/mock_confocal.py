import numpy as np
import json
from lumicks.pylake.detail.image import InfowaveCode
from lumicks.pylake.channel import Continuous, Slice
from lumicks.pylake.kymo import Kymo


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
    enc = json.JSONEncoder()

    axes_metadata = [
        {
            "axis": axis["axis"],
            "cereal_class_version": 1,
            "num of pixels": axis["num of pixels"],
            "pixel size (nm)": axis["pixel size (nm)"],
            "scan time (ms)": 0,
            "scan width (um)": axis["pixel size (nm)"] * axis["num of pixels"] / 1000.0 + 0.5,
        }
        for axis in axes
    ]

    return enc.encode(
        {
            "value0": {
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
            }
        }
    )


def generate_image_data(image, samples_per_pixel, line_padding):
    """Generates the appropriate info_wave and photon_count data for a particular image.

    Parameters
    ----------
    image : array_like
        Image to generate an infowave for.
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

    return np.hstack([generate_infowave_line(line) for line in image.T]), np.hstack(
        [generate_photon_count_line(line) for line in image.T]
    )


class MockConfocalFile:
    def __init__(self, infowave, red_channel=None, blue_channel=None, green_channel=None):
        self.infowave = infowave
        self.red_photon_count = red_channel
        self.green_photon_count = green_channel
        self.blue_photon_count = blue_channel

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
                for pixel_size, axis, num_pixels in zip(pixel_sizes_nm, axes, image.shape)
            ]
        )

        return (
            MockConfocalFile(
                Slice(Continuous(infowave, start=start, dt=dt)),
                Slice(Continuous(photon_counts, start=start, dt=dt)),
            ),
            json.loads(json_string)["value0"],
            start + len(infowave) * dt,
        )


def generate_kymo(name, image, pixel_size_nm, start=4, dt=7, samples_per_pixel=5, line_padding=3):
    confocal_file, json, stop = MockConfocalFile.from_image(
        image,
        pixel_sizes_nm=[pixel_size_nm],
        axes=[0],
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
    )

    return Kymo(name, confocal_file, start, stop, json)
