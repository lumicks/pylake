import numpy as np
import json
import tifffile


class MockTag:
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value


class MockTiffPage:
    def __init__(self, data, start_time, end_time, description="", bit_depth=8):
        self._data = data
        bit_depth = bit_depth if data.ndim == 2 else (bit_depth, bit_depth, bit_depth)
        self.tags = {
            "DateTime": MockTag(f"{start_time}:{end_time}"),
            "ImageDescription": MockTag(description),
            "BitsPerSample": MockTag(bit_depth),
            "SamplesPerPixel": MockTag(1 if (data.ndim == 2) else data.shape[2]),
            "ImageWidth": MockTag(data.shape[1]),
            "ImageLength": MockTag(data.shape[0]),
        }

    def asarray(self):
        return self._data.copy()

    @property
    def description(self):
        return self.tags["ImageDescription"].value


class MockTiffFile:
    def __init__(self, data, times, description="", bit_depth=8):
        self.pages = []
        for d, r in zip(data, times):
            self.pages.append(
                MockTiffPage(d, r[0], r[1], description=description, bit_depth=bit_depth)
            )

    @property
    def num_frames(self):
        return len(self._src.pages)


def make_frame_times(n_frames, step=8, start=10):
    return [[f"{j}", f"{j+step}"] for j in range(start, start + (n_frames + 1) * 100, 10)]


def apply_transform(spots, Tx, Ty, theta, offsets=None):
    theta = np.radians(theta)
    transform_matrix = np.array(
        [[np.cos(theta), -np.sin(theta), Tx], [np.sin(theta), np.cos(theta), Ty], [0, 0, 1]]
    )

    # translate origin by offsets if necessary
    offsets = np.zeros((2, 1)) if offsets is None else np.array(offsets, ndmin=2).T
    spots = spots - offsets
    # reshape spots into coordinate matrix; [x,y,z] as columns
    spots = np.vstack((spots, np.ones(spots.shape[1])))
    # affine transformation
    transformed_spots = np.dot(transform_matrix, spots)[:2]
    # back-translate origin if necessary
    transformed_spots = transformed_spots + offsets

    return transform_matrix, transformed_spots


def make_image(spots, bit_depth):
    # RGB image, 2D (normalized) gaussians at spot locations
    sigma = np.eye(2) * 5
    X, Y = np.meshgrid(np.arange(0, 200), np.arange(0, 100))
    img = np.zeros(X.shape)

    for x, y in spots.T:
        mu = np.array([x, y])[:, np.newaxis]
        XX = np.vstack((X.ravel(), Y.ravel())) - mu
        quad_form = np.sum(np.dot(XX.T, np.linalg.inv(sigma)) * XX.T, axis=1)
        Z = np.exp(-0.5 * quad_form)
        img += Z.reshape(X.shape)
    img = img / img.max()

    return (img * (2 ** bit_depth - 1)).astype(f"uint{bit_depth}")


def _make_base_description(version, bit_depth):
    # version == 1 corresponds to metadata generated by Bluelake v1.7.0-beta1.c
    # version == 2 corresponds to metadata generated by Bluelake v1.7.0

    laser_on = lambda c: f"{c} Excitation Laser on"
    laser_wavelength = lambda c: f"{c} Excitation Laser wavelength (nm)"
    laser_power = lambda c, suff: f"{c} Excitation Laser power {suff}(%)"
    colors = ("Blue", "Green", "Red")

    description = {
        "Background subtraction": None,
        "Exposure time (ms)": None,
        "Focus lock": None,
        "Frame averaging": None,
        "Frame rate (Hz)": None,
        "Pixel clock (MHz)": None,
        "Region of interest (x, y, width, height)": [0, 0, 200, 100],
    }
    for c in colors:
        description[laser_wavelength(c)] = None
        description[laser_power(c, "" if version == 1 else "level ")] = None
    if version > 1:
        for c in colors:
            description[laser_on(c)] = None
        description["Bit depth"] = bit_depth
        description["Exposure sync available"] = None
        description["Exposure sync enabled"] = None
    return description


def make_irm_description(version, bit_depth):
    description = _make_base_description(version, bit_depth)
    description["Camera"] = "IRM"
    return description


def make_wt_description(version, bit_depth, m_red, m_blue, offsets):
    if version == 1:
        alignment_matrices = lambda color: f"Alignment {color} channel"
        channel_choices = ("red", "green", "blue")
    else:
        alignment_matrices = lambda index: f"Channel {index} alignment"
        channel_choices = range(3)

    offsets = [0, 0] if offsets is None else offsets
    matrices = (m_red, np.eye(3), m_blue)

    description = _make_base_description(version, bit_depth)
    description["Camera"] = "WT"
    for c, mat in zip(channel_choices, matrices):
        description[alignment_matrices(c)] = mat[:2].ravel().tolist()
    description["Alignment region of interest (x, y, width, height)"] = [*offsets, 200, 100]
    description["TIRF"] = None
    description["TIRF angle (device units)"] = None
    return description


def make_alignment_image_data(
    spots,
    red_warp_parameters,
    blue_warp_parameters,
    bit_depth,
    offsets=None,
    camera="wt",
    version=1,
):

    spots = np.array(spots).T  # [2 x N]
    m_red, red_spots = apply_transform(spots, offsets=offsets, **red_warp_parameters)
    m_blue, blue_spots = apply_transform(spots, offsets=offsets, **blue_warp_parameters)

    red_image = make_image(red_spots, bit_depth)
    green_image = make_image(spots, bit_depth)
    blue_image = make_image(blue_spots, bit_depth)

    reference_image = np.repeat(green_image[:, :, np.newaxis], 3, axis=2)
    warped_image = np.stack((red_image, green_image, blue_image), axis=2).squeeze()
    if camera == "wt":
        description = make_wt_description(version, bit_depth, m_red, m_blue, offsets)
    elif camera == "irm":
        description = make_irm_description(version, bit_depth)
        # IRM images are grayscale so they only have 1 channel
        reference_image = reference_image[:, :, 1]
        warped_image = warped_image[:, :, 1]
    else:
        raise ValueError("camera argument must be 'wt' or 'irm'")

    return reference_image, warped_image, description, bit_depth


def write_tiff_file(image, description, n_frames, filename):
    # We use the dimension of image data to evaluate the number of color channels
    channels = 1 if image.ndim == 2 else 3
    movie = np.stack([image for n in range(n_frames)], axis=0)

    # Orientation = ORIENTATION.TOPLEFT
    tag_orientation = (274, "H", 1, 1, False)
    # SampleFormat = SAMPLEFORMAT.UINT
    tag_sample_format = (339, "H", channels, (1,) * channels, False)

    with tifffile.TiffWriter(filename) as tif:
        for n, frame in enumerate(movie):
            str_datetime = f"{n*10+10}:{n*10+18}"
            tag_datetime = (306, "s", len(str_datetime), str_datetime, False)
            tif.write(
                frame,
                description=json.dumps(description, indent=4),
                software="Bluelake Unknown",
                metadata=None,
                contiguous=False,
                extratags=(tag_orientation, tag_sample_format, tag_datetime),
            )
