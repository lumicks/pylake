import numpy as np
import re
import json
import cv2
import tifffile
import warnings
import enum
from copy import copy
from dataclasses import dataclass


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

    def __init__(self, page, description):
        self._page = page
        self._description = description

    @property
    def _is_aligned(self):
        return self._description._alignment.is_aligned

    def _align_image(self):
        """reconstruct image using alignment matrices from Bluelake; return aligned image as a NumPy array"""
        align_mats = [self._description.alignment_matrix(color) for color in ("red", "blue")]

        img = self.raw_data
        rows, cols, _ = img.shape
        for mat, channel in zip(align_mats, (0, 2)):
            img[:, :, channel] = cv2.warpAffine(
                img[:, :, channel],
                mat,
                (cols, rows),
                flags=(cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return img

    @property
    def data(self):
        return (
            self._align_image()
            if self._description._alignment.status == AlignmentStatus.ready
            else self._page.asarray()
        )

    @property
    def raw_data(self):
        return self._page.asarray()

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

    def __init__(self, tiff_file, align_requested):
        self._tiff_file = tiff_file
        self._description = ImageDescription(tiff_file, align_requested)

        # warn on file open if alignment is requested, but not possible
        # stacklevel=4 corresponds to CorrelatedStack.__init__()
        if self._description._alignment.has_problem:
            warnings.warn(self._description._alignment.status.value, stacklevel=4)

    def get_frame(self, frame):
        return TiffFrame(self._tiff_file.pages[frame], description=self._description)

    @staticmethod
    def from_file(image_file, align_requested):
        return TiffStack(tifffile.TiffFile(image_file), align_requested=align_requested)

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
    _cmap : dict
        dictionary of color : channel index pairs
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
            self._cmap = {}
            return

        # update format if necessary
        self._cmap = {"red": 0, "green": 1, "blue": 2}
        if "Alignment red channel" in self.json:
            for color, j in self._cmap.items():
                self.json[f"Channel {j} alignment"] = self.json.pop(f"Alignment {color} channel")
                self.json[f"Channel {j} detection wavelength (nm)"] = "N/A"

        # check alignment status
        if not self.is_rgb:
            self._alignment = Alignment(align_requested, AlignmentStatus.not_applicable, False)
        elif "Channel 0 alignment" in self.json.keys():
            self._alignment = Alignment(align_requested, AlignmentStatus.ready, True)
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

    def _raw_alignment_matrix(self, color):
        return np.array(self.json[f"Channel {self._cmap[color]} alignment"]).reshape((2, 3))

    def alignment_matrix(self, color):
        def correct_alignment_offset(alignment, x_offset, y_offset):
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
            return np.dot(back_translation, np.dot(original, translation))[:2, :]

        align_mat = self._raw_alignment_matrix(color)
        x_offset, y_offset = self.offsets
        if x_offset == 0 and y_offset == 0:
            return align_mat
        else:
            return correct_alignment_offset(align_mat, x_offset, y_offset)

    @property
    def for_export(self):
        out = copy(self.json)
        if self._alignment.status == AlignmentStatus.ready:
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
