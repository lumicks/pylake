import re
import enum
import json
import warnings
from copy import copy
from dataclasses import dataclass

import numpy as np
import tifffile

from .plotting import parse_color_channel
from ..adjustments import no_adjustment


def _get_page_timestamps(page):
    """Get the starting timestamp from a TIFF page"""
    timestamp_string = re.search(r"^(\d+):(\d+)$", page.tags["DateTime"].value)
    if timestamp_string:
        return np.int64(timestamp_string.group(1)), np.int64(timestamp_string.group(2))
    else:
        raise ValueError("Incorrectly formatted timestamp metadata")


class TiffFrame:
    """Thin wrapper around a TIFF frame stack. For camera videos timestamps are stored in the DateTime tag in
    the format start:end.

    Parameters
    ----------
    page : tifffile.tifffile.TiffPage
        Tiff page recorded from a camera in Bluelake.
    description : ImageDescription
        Wrapper around TIFF 'ImageDescription' metadata tag.
    roi : Roi
        Region of interest corner coordinates.
    tether : Tether
        Tether coordinates.
    """

    def __init__(self, page, description, roi, tether):
        self._page = page
        self._description = description
        self._roi = roi
        self._tether = tether

    @property
    def _is_aligned(self):
        return self._description._alignment.is_aligned

    def _normalize_array(self):
        img = self._page.asarray()
        if not self.is_rgb or img.shape[-1] == 3:
            return img

        # Handle two color images. We up-convert them to a 3-color image to be able to handle
        # rgb-mappable images in a uniform manner.
        three_colors = np.zeros((img.shape[0], img.shape[1], 3))
        three_colors[:, :, self._description.channel_order] = img
        return three_colors

    def _align_image(self):
        """reconstruct image using alignment matrices from Bluelake; return aligned image as a NumPy array"""
        img = self._normalize_array()
        for channel, mat in self._description._alignment_matrices.items():
            mat = self._tether.rot_matrix * mat
            img[:, :, channel] = mat.warp_image(img[:, :, channel])
        return img

    @property
    def data(self):
        data = (
            self._align_image()
            if self._description._alignment.do_alignment
            else self._tether.rot_matrix.warp_image(self._normalize_array())
        )
        return self._roi(data)

    @property
    def raw_data(self):
        return self._roi(self._normalize_array())

    @property
    def bit_depth(self):
        bit_depth = self._page.tags["BitsPerSample"].value
        if self.is_rgb:  # (int r, int g, int b)
            return bit_depth[0]
        else:  # int
            return bit_depth

    @property
    def is_rgb(self):
        return self._description.is_rgb

    def _get_plot_data(self, channel="rgb", adjustment=no_adjustment):
        """Return data as a numpy array, appropriate for use by `imshow`.

        Parameters
        ----------
        channel : str
            Which channel to return. Options are: "red", "green", "blue" or "rgb".
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.

        Returns
        -------
        if data is grayscale or channel in ('red', 'green', 'blue')
            return data as is
        if channel is 'rgb',  converted to float in range [0,1]
        """

        if not self.is_rgb:
            return self.data

        channel = parse_color_channel(channel)

        if channel in ("r", "g", "b"):
            return self.data[:, :, "rgb".index(channel)]

        data = self.data.astype(float)

        return adjustment._get_data_rgb(data, channel=channel)

    @property
    def start(self):
        return self.frame_timestamp_range[0]

    @property
    def stop(self):
        return self.exposure_timestamp_range[1]

    @property
    def frame_timestamp_range(self):
        return _get_page_timestamps(self._page)

    @property
    def exposure_timestamp_range(self):
        try:
            json_metadata = json.loads(self._page.description)
        except json.decoder.JSONDecodeError:
            json_metadata = {}

        start, stop = _get_page_timestamps(self._page)
        stop = (
            start + int(np.round(1e6 * json_metadata["Exposure time (ms)"]))
            if "Exposure time (ms)" in json_metadata
            else stop
        )

        return start, stop


class WrappedTiffFile:
    """TIFF file wrapper that closes file upon garbage collection. The reason we can't put this behavior in ImageStack
    or TiffStack itself is because TiffFiles are shared between instances of these."""

    def __init__(self, file, *args, **kwargs):
        self._src = tifffile.TiffFile(file, *args, **kwargs)
        self._fn = self._src.filename

    @property
    def pages(self):
        if not self._src:
            raise IOError(
                f"The file handle for this TiffStack ({self._fn}) has already been closed."
            )

        return self._src.pages

    def close(self):
        if self._src:
            self._src.close()

        self._src = None

    def __del__(self):
        self.close()


class TiffStack:
    """TIFF images exported from Bluelake

    Parameters
    ----------
    tiff_files : list of WrappedTiffFile
        List of TIFF files recorded from a camera in Bluelake.
    align_requested : bool
        Whether color channel alignment is requested.

    Raises
    ------
    ValueError
        If any of the provided TIFF are not compatible. To be compatible, they both need to be RGB
        or non-RGB. They need to have the same width and height and they need to have the same
        alignment matrices.
    """

    def __init__(self, tiff_files, align_requested, roi=None, tether=None):
        # Make sure the timestamps are in order (since downstream functionality depends on this)
        try:
            timestamps = [_get_page_timestamps(file.pages[0])[0] for file in tiff_files]
        except (KeyError, ValueError):  # Missing (KeyError) or incorrectly formatted (ValueError)
            raise RuntimeError("The timestamp data was incorrectly formatted")

        order = np.argsort(timestamps)
        self._tiff_files = [tiff_files[idx] for idx in order]

        descriptions = [ImageDescription(tiff_file, align_requested) for tiff_file in tiff_files]

        # Verify that the images in the stack are compatible accepting it as one stack,
        # this means that they need to have the same transform, number of pixels and channels.
        for description in descriptions:
            descriptions[0].verify_stack_similarity(description)

        self._description = descriptions[0]

        # warn on file open if alignment is requested, but not possible
        # stacklevel=4 corresponds to ImageStack.__init__()
        if self._description._alignment.has_problem:
            warnings.warn(self._description._alignment.status.value, stacklevel=4)

        if roi is None:
            self._roi = Roi(0, self._description.width, 0, self._description.height)
        else:
            self._roi = roi

        self._tether = Tether(self._roi.origin) if tether is None else tether

    @property
    def is_rgb(self):
        return self._description.is_rgb

    @property
    def _num_frames_per_tiff(self):
        return [len(tiff_file.pages) for tiff_file in self._tiff_files]

    def get_frame(self, frame):
        cumulative_len = np.cumsum(np.hstack((0, self._num_frames_per_tiff)))
        file_idx = np.argmax(frame < cumulative_len) - 1
        frame_within_tiff = frame - cumulative_len[file_idx]

        return TiffFrame(
            self._tiff_files[file_idx].pages[frame_within_tiff],
            description=self._description,
            roi=self._roi,
            tether=self._tether,
        )

    @staticmethod
    def from_file(image_files, align_requested):
        """Construct TiffStack from file(s)

        Parameters
        ----------
        image_files : list of str or str
            Tiff file(s) to read
        align_requested : bool
            Does the user request these images to be aligned?
        """
        file_names = image_files if isinstance(image_files, (list, tuple)) else [image_files]

        return TiffStack(
            [WrappedTiffFile(fn) for fn in file_names], align_requested=align_requested
        )

    def with_roi(self, roi):
        """Define a region of interest (ROI) to crop from raw image.

        Parameters
        ----------
        roi : (int, int, int, int)
            (x_min, x_max, y_min, y_max) pixel coordinates defining the ROI.

        """
        roi = self._roi.crop(roi)
        tether = self._tether.with_new_offsets(roi.origin)

        return TiffStack(
            self._tiff_files,
            self._description._alignment.requested,
            roi=roi,
            tether=tether,
        )

    def with_tether(self, points):
        """Define the endpoints of the tether and rotate the image such that the tether is horizontal.

        The rotation angle is calculated from the slope of the line defined by the two endpoints
        while the center of rotation is defined as the centroid of the two points.

        Parameters
        ----------
        point_1 : (float, float)
            (x, y) coordinates of the tether start point
        point_2 : (float, float)
            (x, y) coordinates of the tether end point
        """
        # if tether is already defined, un-rotate input points
        if self._tether:
            mat = self._tether.offsets.invert() * (
                self._tether.rot_matrix.invert() * self._tether.offsets
            )
            points = mat.warp_coordinates(points)

        return TiffStack(
            self._tiff_files,
            self._description._alignment.requested,
            roi=self._roi,
            tether=Tether(self._roi.origin, points),
        )

    @property
    def _shape(self):
        """Return cropped image shape as (n_rows, n_columns)."""
        return self._roi.shape

    @property
    def num_frames(self):
        return np.sum(self._num_frames_per_tiff)

    def close(self):
        for file in self._tiff_files:
            file.close()


def _parse_wavelength(wavelength_field):
    """Parse excitation wavelength"""
    try:
        return float(wavelength_field.split("/")[0])  # Format is wavelength / width
    except ValueError:
        return


class ImageDescription:
    """Wrapper around TIFF ImageDescription tag and other pylake specific metadata.

    Parameters
    ----------
    tiff_file : tifffile.TiffFile
        TIFF file recorded from a camera in Bluelake.
    align_requested : bool
        whether user has requested color channels to be aligned via affine transform.

    Attributes
    ----------
    is_rgb : bool
        whether data is single-channel or RGB
    _alignment : Alignment
        details of color channel alignment.
    json : dict
        dictionary of parsed json string held in frame ImageDescription tag.
    """

    def __init__(self, tiff_file, align_requested):
        first_page = tiff_file.pages[0]
        tags = first_page.tags
        self.is_rgb = tags["SamplesPerPixel"].value > 1
        self.width = tags["ImageWidth"].value
        self.height = tags["ImageLength"].value
        self.software = tags["Software"].value if "Software" in tags else ""
        self.pixelsize_um = None
        self._alignment_matrices = {}

        # parse json string stored in ImageDescription tag
        try:
            self.json = json.loads(first_page.description)
        except json.decoder.JSONDecodeError:
            self.json = {}

        # if metadata is missing, set default values
        if len(self.json) == 0:
            self._alignment = Alignment(align_requested, AlignmentStatus.empty, False)
            return

        self.pixelsize_um = (
            float(self.json["Pixel calibration (nm/pix)"]) / 1000
            if "Pixel calibration (nm/pix)" in self.json
            else None
        )

        # update json fields if necessary
        if "Alignment red channel" in self.json:
            for j, color in enumerate(("red", "green", "blue")):
                self.json[f"Channel {j} alignment"] = self.json.pop(f"Alignment {color} channel")
                self.json[f"Channel {j} detection wavelength (nm)"] = "N/A"

        excitation_colors = [
            key.split(" ")[0] for key in self.json.keys() if "Excitation Laser wavelength" in key
        ]

        self.channel_order = (0, 1, 2)
        if len(excitation_colors) == 2:
            self.channel_order = tuple(
                ix
                for ix, color in enumerate(("Red", "Green", "Blue"))
                if color in excitation_colors
            )

        excitation_filter_wavelengths = [
            _parse_wavelength(self.json[f"Channel {channel} detection wavelength (nm)"])
            for channel in range(3)
            if f"Channel {channel} detection wavelength (nm)" in self.json
        ]

        if excitation_filter_wavelengths and all(excitation_filter_wavelengths):
            # Validate that the wavelengths are in ascending order
            if not all(np.diff(excitation_filter_wavelengths) < 0):
                raise RuntimeError(
                    f"Wavelengths are not in descending order {excitation_filter_wavelengths}"
                )

        self._alignment_matrices = {
            rgb_channel: TransformMatrix.from_alignment(
                self._raw_alignment_matrix(channel), *self.offsets
            ).invert()
            for channel, rgb_channel in enumerate(self.channel_order)
            if f"Channel {channel} alignment" in self.json
        }

        # check alignment status
        if not self.is_rgb:
            self._alignment = Alignment(align_requested, AlignmentStatus.not_applicable, False)
        elif "Channel 0 alignment" in self.json.keys():
            self._alignment = Alignment(align_requested, AlignmentStatus.ready, align_requested)
        elif any([re.search(r"^Applied (.*)channel(.*)$", key) for key in self.json.keys()]):
            self._alignment = Alignment(align_requested, AlignmentStatus.applied, True)
        else:
            self._alignment = Alignment(align_requested, AlignmentStatus.missing, False)

    def verify_stack_similarity(self, other):
        """Verifies that the metadata for these images reflects a compatible image

        Parameters
        ----------
        other : ImageDescription
            Image metadata

        Raises
        ------
        ValueError
            If the two `ImageDescription` are not compatible. To be compatible, they both need to
            be RGB or non-RGB. They need to have the same width and height and they need to have
            the same alignment matrices.
        """

        if self.is_rgb != other.is_rgb:
            raise ValueError("Cannot mix RGB and non-RGB stacks.")

        if self.width != other.width or self.height != other.height:
            raise ValueError("Cannot mix differently sized tiffs into a single stack.")

        if self._legacy_exposure != other._legacy_exposure:
            raise ValueError("Cannot mix tiffs exported before and after Pylake v1.3.2.")

        # We only allow merging stacks with the exact same alignment settings
        if self._alignment_matrices or other._alignment_matrices:
            if self._alignment.do_alignment != other._alignment.do_alignment:
                raise ValueError("Alignment matrices must be the same for stacks to be merged.")

            # Checks whether they both have alignment matrices and whether they are the same
            diff_keys = set(self._alignment_matrices.keys()) - set(other._alignment_matrices.keys())
            if diff_keys:
                raise ValueError(
                    "Alignment matrices must be the same for stacks to be merged. The following "
                    f"alignment matrices were found in one stack but not the other {diff_keys}."
                )

            # Check the actual matrices we have here
            for channel, mat in self._alignment_matrices.items():
                if mat != other._alignment_matrices[channel]:
                    raise ValueError(
                        "Alignment matrices must be the same for stacks to be merged. The "
                        f"alignment matrix for channel {channel} is different."
                    )

    @property
    def _legacy_exposure(self):
        return "Pylake" in self.software and "Exposure time (ms)" not in self.json

    @property
    def alignment_roi(self):
        return np.array(self.json["Alignment region of interest (x, y, width, height)"])

    @property
    def roi(self):
        return np.array(self.json["Region of interest (x, y, width, height)"])

    @property
    def offsets(self):
        return self.alignment_roi[:2] - self.roi[:2]

    def _raw_alignment_matrix(self, index):
        return np.array(self.json[f"Channel {index} alignment"]).reshape((2, 3))

    @property
    def for_export(self):
        out = copy(self.json)
        if self._alignment.do_alignment:
            for j in range(3):
                if f"Channel {j} alignment" in out:
                    out[f"Applied channel {j} alignment"] = out.pop(f"Channel {j} alignment")

        # Add some Pylake specific metadata
        out["Pylake"] = {
            "Channel mapping": {
                ("red", "green", "blue")[channel_idx]: idx
                for idx, channel_idx in enumerate(self.channel_order)
            }
        }
        return out


class AlignmentStatus(enum.Enum):
    empty = "File does not contain metadata. Only raw data is available"
    missing = "File does not contain alignment matrices. Only raw data is available"
    ready = "alignment is possible"
    not_applicable = "single channel data, alignment not applicable"
    applied = "alignment has already been applied, do not re-align"


@dataclass
class Alignment:
    requested: bool
    status: AlignmentStatus
    is_aligned: bool

    @property
    def has_problem(self):
        """True if alignment is requested but cannot be performed."""
        bad_status = self.status in (AlignmentStatus.empty, AlignmentStatus.missing)
        return self.requested and bad_status

    @property
    def do_alignment(self):
        """Should alignment be performed?"""
        return self.status == AlignmentStatus.ready and self.requested


@dataclass
class Roi:
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    def __post_init__(self):
        if np.any(self.asarray() < 0):
            raise ValueError("Pixel indices must be >= 0.")
        if (self.x_max <= self.x_min) or (self.y_max <= self.y_min):
            raise ValueError("Max must be larger than min.")

    def __call__(self, data):
        return data[self.y_min : self.y_max, self.x_min : self.x_max]

    def asarray(self):
        """Return corner coordinates as `np.array`."""
        return np.array([self.x_min, self.x_max, self.y_min, self.y_max])

    def crop(self, roi):
        """Crop again, taking into account origin of current ROI."""
        roi = np.array(
            [
                pos if pos is not None else default
                for pos, default in zip(roi, (0, self.shape[1], 0, self.shape[0]))
            ]
        )

        # Support negative indexing
        roi_max = np.asarray([self.shape[x] for x in (1, 1, 0, 0)])
        negative_indices = roi < 0
        roi[negative_indices] += roi_max[negative_indices]

        # Clip to image
        roi = [np.clip(x, 0, dim_max) for (x, dim_max) in zip(roi, roi_max)]

        roi = np.array(roi) + [self.x_min, self.x_min, self.y_min, self.y_min]
        return Roi(*roi)

    @property
    def shape(self):
        return self.y_max - self.y_min, self.x_max - self.x_min

    @property
    def origin(self):
        return np.array([self.x_min, self.y_min])


class TransformMatrix:
    def __init__(self, matrix=None):
        """Wrapper around affine transformation matrix with convenience functions."""
        self.matrix = np.eye(3) if matrix is None else np.vstack((matrix, [0, 0, 1]))

    def __mul__(self, mat):
        """Perform matrix multiplication such that `self * mat == np.matmul(self, mat)`."""
        if not isinstance(mat, TransformMatrix):
            raise TypeError(f"Operands must be of type `TransformMatrix`, got {type(mat)}.")
        return TransformMatrix(np.matmul(self.matrix, mat.matrix)[:2])

    def __eq__(self, other):
        return np.all(self.matrix == other.matrix)

    def invert(self):
        return TransformMatrix(np.linalg.inv(self.matrix)[:2])

    @classmethod
    def from_alignment(cls, alignment, x_offset=0, y_offset=0):
        """Recalculate matrix with offset. Same implementation as Bluelake.

        Parameters
        ----------
        alignment: np.ndarray
            alignment matrix read from Bluelake metadata
        x_offset: float
            alignment ROI x offset
        y_offset: float
            alignment ROI y offset
        """
        if x_offset == 0 and y_offset == 0:
            return cls(alignment)

        # translate the origin of the image so that it matches that of the original transform
        translation = np.eye(3)
        translation[0, -1] = -x_offset
        translation[1, -1] = -y_offset
        # apply the original transform to the translated image.
        # it only needs to be resized from a 2x3 to a 3x3 matrix
        original = np.vstack((alignment, [0, 0, 1]))
        # translate the image back to the original origin.
        # takes into account both the offset and the scaling performed by the first step
        back_translation = np.eye(3)
        back_translation[0, -1] = original[0, 0] * x_offset
        back_translation[1, -1] = original[1, 1] * y_offset
        # concatenate the transforms by multiplying their matrices and ignore the unused 3rd row
        return cls(np.dot(back_translation, np.dot(original, translation))[:2, :])

    @classmethod
    def rotation(cls, theta, center):
        """Construct matrix for rotation by angle `theta` about a point `center`.

        Parameters
        ----------
        theta: float
            angle of rotation in degrees (counter-clockwise)
        center: np.ndarray
            (x, y) point, center of rotation
        """
        from skimage.transform import EuclideanTransform

        tf_shift = EuclideanTransform(translation=tuple(-c for c in center))
        tf_rotate = EuclideanTransform(rotation=-np.deg2rad(theta))
        tf_shift_back = EuclideanTransform(translation=center)
        rotation = tf_shift + tf_rotate + tf_shift_back
        return cls(rotation.params[:-1, :])

    @classmethod
    def translation(cls, x, y):
        """Construct matrix for translation by `x` and `y`.

        Parameters
        ----------
        x: float
            translation in x direction
        y: float
            translation in y direction
        """
        matrix = np.eye(3)
        matrix[:2, -1] = (x, y)
        return cls(matrix[:2])

    def warp_image(self, data):
        """Apply affine transformation to image data.

        Parameters
        ----------
        data: np.ndarray
            image data
        """
        from skimage import transform

        if np.all(np.equal(self.matrix, np.eye(3))):
            return data

        return transform.warp(
            data,
            self.invert().matrix,
            order=1,  # Linear interpolation
            mode="constant",
            cval=0.0,  # borderValue
            clip=True,
            preserve_range=True,
        )

    def warp_coordinates(self, coordinates):
        """Apply affine transformation to coordinate points.

        Parameters
        ----------
        coordinates: list
            list of (x, y) coordinates
        """
        if not np.any(coordinates):
            return coordinates

        coordinates = np.vstack(coordinates).T
        coordinates = np.vstack((coordinates, np.ones(coordinates.shape[1])))

        warped = np.matmul(self.matrix, coordinates)[:2].T
        return [(x, y) for x, y in warped]


class Tether:
    def __init__(self, offsets, points=None):
        """Helper class to define the coordinates of the tether ends.

        Parameters
        ----------
        offsets: np.ndarray
            (x,y) coordinates of the ROI origin
        points: list
            list of (x,y) coordinates for the tether end points
            in the coordinate system of the current processed image

        Attributes
        ----------
        offsets: np.ndarray
            (x,y) coordinate offsets from the raw image origin
        rot_matrix: TransformMatrix
            rotation matrix to be applied to the raw image data
            to make the tether horizontal in the final image
        """
        self.offsets = TransformMatrix.translation(*offsets)

        if points is None:
            self._ends = None
            self.rot_matrix = TransformMatrix()
        else:
            # list of (x,y) coordinates for the tether end points
            # in the coordinate system of the raw image data
            self._ends = self.offsets.warp_coordinates(points)
            # calculate rotation matrix
            dy, dx = self._ends[1][1] - self._ends[0][1], self._ends[1][0] - self._ends[0][0]
            theta = np.degrees(np.arctan2(dy, dx))
            center = np.mean(np.vstack(self._ends), axis=0)
            self.rot_matrix = TransformMatrix.rotation(theta, center)

    def with_new_offsets(self, new_offsets):
        if not self:  # empty tether
            return Tether(new_offsets)
        # find end points in new ROI coordinate system, reform tether
        new_tether_ends = self._ends - np.array(new_offsets)
        return Tether(new_offsets, new_tether_ends)

    def __bool__(self):
        return self._ends is not None

    @property
    def ends(self):
        """Tether end points in the processed image coordinate system."""
        rotated_ends = self.rot_matrix.warp_coordinates(self._ends)
        return self.offsets.invert().warp_coordinates(rotated_ends)


def _frame_timestamps_from_exposure_timestamps(ts_ranges):
    """Generate frame timestamp ranges from confocal scans exported with legacy Pylake metadata
    format.

    For confocal Scans exported with Pylake `<v1.3.2` timestamps contained the start and end time
    of the exposure, rather than the full frame. This behavior was inconsistent with Bluelake
    and therefore changed. To be able to load files generated in this way, we need to be able
    to reconstruct timestamps using only the start timestamp of each frame

    Parameters
    ----------
    ts_ranges : list of tuple
        Modern timestamp ranges in nanoseconds given by a list of tuples according to the format
        (start of frame, start of next frame).
    """
    frame_ts = [(leading[0], trailing[0]) for leading, trailing in zip(ts_ranges, ts_ranges[1:])]
    if len(ts_ranges) >= 2:
        dt = ts_ranges[-1][0] - ts_ranges[-2][0]
        stop = ts_ranges[-1][0] + dt
    else:
        stop = ts_ranges[-1][1]
    frame_ts.append((ts_ranges[-1][0], stop))

    return frame_ts
