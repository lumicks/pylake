import numpy as np
import re
import json
import cv2
import tifffile
import warnings
import enum
from copy import copy
from dataclasses import dataclass, field


class TiffFrame:
    """Thin wrapper around a TIFF frame stack. For camera videos timestamps are stored in the DateTime tag in
    the format start:end.

    Parameters
    ----------
    page : tifffile.tifffile.TiffPage
        Tiff page recorded from a camera in Bluelake.
    description : ImageDescription
        Wrapper around TIFF 'ImageDescription' metadata tag.
    """

    def __init__(self, page, description, roi):
        self._page = page
        self._description = description
        self._roi = roi

    @property
    def _is_aligned(self):
        return self._description._alignment.is_aligned

    def _align_image(self):
        """reconstruct image using alignment matrices from Bluelake; return aligned image as a NumPy array"""
        img = self._page.asarray()
        for channel, mat in self._description._alignment_matrices.items():
            img[:, :, channel] = mat.warp_image(img[:, :, channel])
        return img

    @property
    def data(self):
        data = (
            self._align_image()
            if self._description._alignment.do_alignment
            else self._page.asarray()
        )
        return self._roi(data)

    @property
    def raw_data(self):
        return self._roi(self._page.asarray())

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

    def _get_plot_data(self, channel="rgb", vmax=None):
        """return data an numpy array, appropriate for use by `imshow`
        if data is grayscale or channel in ('red', 'green', 'blue')
            return data as is
        if channel is 'rgb',  converted to float in range [0,1] and correct for optional vmax argument:
            None  : normalize data to max signal of all channels
            float : normalize data to vmax value
        """

        if not self.is_rgb:
            return self.data

        if channel.lower() == "rgb":
            data = (self.data / (2 ** self.bit_depth - 1)).astype(float)
            if vmax is None:
                return data / data.max()
            else:
                return data / vmax
        else:
            try:
                return self.data[:, :, ("red", "green", "blue").index(channel.lower())]
            except ValueError:
                raise ValueError(f"'{channel}' is not a recognized channel")

    @property
    def start(self):
        timestamp_string = re.search(r"^(\d+):\d+$", self._page.tags["DateTime"].value)
        return np.int64(timestamp_string.group(1)) if timestamp_string else None

    @property
    def stop(self):
        timestamp_string = re.search(r"^\d+:(\d+)$", self._page.tags["DateTime"].value)
        return np.int64(timestamp_string.group(1)) if timestamp_string else None


class TiffStack:
    """TIFF images exported from Bluelake

    Parameters
    ----------
    tiff_file : tifffile.TiffFile
        TIFF file recorded from a camera in Bluelake.
    align_requested : bool
        Whether color channel alignment is requested.
    """

    def __init__(self, tiff_file, align_requested, roi=None):
        self._tiff_file = tiff_file
        self._description = ImageDescription(tiff_file, align_requested)

        # warn on file open if alignment is requested, but not possible
        # stacklevel=4 corresponds to CorrelatedStack.__init__()
        if self._description._alignment.has_problem:
            warnings.warn(self._description._alignment.status.value, stacklevel=4)

        if roi is None:
            height, width = self._src_shape
            self._roi = Roi(0, width, 0, height)
        else:
            self._roi = roi

    def get_frame(self, frame):
        return TiffFrame(self._tiff_file.pages[frame], description=self._description, roi=self._roi)

    @staticmethod
    def from_file(image_file, align_requested):
        return TiffStack(tifffile.TiffFile(image_file), align_requested=align_requested)

    def with_roi(self, roi):
        return TiffStack(
            self._tiff_file,
            self._description._alignment.requested,
            roi=self._roi.crop(roi),
        )

    @property
    def _src_shape(self):
        """Return source image shape as (n_rows, n_columns)."""
        width = self._tags["ImageWidth"].value
        height = self._tags["ImageLength"].value
        return (height, width)

    @property
    def _shape(self):
        """Return cropped image shape as (n_rows, n_columns)."""
        return self._roi.shape

    @property
    def _tags(self):
        return self._tiff_file.pages[0].tags

    @property
    def num_frames(self):
        return len(self._tiff_file.pages)


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
        self.is_rgb = first_page.tags["SamplesPerPixel"].value == 3

        # parse json string stored in ImageDescription tag
        try:
            self.json = json.loads(first_page.description)
        except json.decoder.JSONDecodeError:
            self.json = {}

        # if metadata is missing, set default values
        if len(self.json) == 0:
            self._alignment = Alignment(align_requested, AlignmentStatus.empty, False)
            return

        # update json fields if necessary
        if "Alignment red channel" in self.json:
            for j, color in enumerate(("red", "green", "blue")):
                self.json[f"Channel {j} alignment"] = self.json.pop(f"Alignment {color} channel")
                self.json[f"Channel {j} detection wavelength (nm)"] = "N/A"

        if "Channel 0 alignment" in self.json:
            self._alignment_matrices = {
                channel: TransformMatrix.from_alignment(
                    self._raw_alignment_matrix(channel), *self.offsets
                ).invert()
                for channel in range(3)
            }
        else:
            self._alignment_matrices = {}

        # check alignment status
        if not self.is_rgb:
            self._alignment = Alignment(align_requested, AlignmentStatus.not_applicable, False)
        elif "Channel 0 alignment" in self.json.keys():
            self._alignment = Alignment(align_requested, AlignmentStatus.ready, align_requested)
        elif any([re.search(r"^Applied (.*)channel(.*)$", key) for key in self.json.keys()]):
            self._alignment = Alignment(align_requested, AlignmentStatus.applied, True)
        else:
            self._alignment = Alignment(align_requested, AlignmentStatus.missing, False)

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
                out[f"Applied channel {j} alignment"] = out.pop(f"Channel {j} alignment")
        return json.dumps(out, indent=4)


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
        if roi[1] > self.shape[1] or roi[3] > self.shape[0]:
            raise ValueError("Pixel indices cannot exceed image size.")
        roi = np.array(roi) + [self.x_min, self.x_min, self.y_min, self.y_min]
        return Roi(*roi)

    @property
    def shape(self):
        return self.y_max - self.y_min, self.x_max - self.x_min


class TransformMatrix:
    def __init__(self, matrix=None):
        """Wrapper around affine transformation matrix with convenience functions."""
        self.matrix = np.eye(3) if matrix is None else np.vstack((matrix, [0, 0, 1]))

    def __mul__(self, mat):
        """Perform matrix multiplication such that `self * mat == np.matmul(self, mat)`."""
        assert isinstance(mat, TransformMatrix), "Operands must be of type `TransformMatrix`."
        return TransformMatrix(np.matmul(self.matrix, mat.matrix)[:2])

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
            angle of rotation
        center: np.ndarray
            (x, y) point, center of rotation
        """
        return cls(cv2.getRotationMatrix2D(center, theta, scale=1.0))

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

        if np.all(np.equal(self.matrix, np.eye(3))):
            return data

        data = np.atleast_3d(data)
        rows, cols, channels = data.shape
        image = [
            cv2.warpAffine(
                data[:, :, channel],
                self.matrix[:2],
                (cols, rows),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            for channel in range(channels)
        ]
        return np.stack(image, axis=2).squeeze()

    def warp_coordinates(self, coordinates):
        """Apply affine transformation to coordinate points.

        Parameters
        ----------
        coordinates: list
            list of (x, y) coordinates
        """
        coordinates = np.vstack(coordinates).T
        coordinates = np.vstack((coordinates, np.ones(coordinates.shape[1])))

        warped = np.matmul(self.matrix, coordinates)[:2].T
        return [(x, y) for x, y in warped]
