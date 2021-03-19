import numpy as np
import re
import json
import cv2
import tifffile
import warnings
from copy import copy


class TiffFrame:
    """Thin wrapper around a TIFF frame stack. For camera videos timestamps are stored in the DateTime tag in
    the format start:end.

    Parameters
    ----------
    page : tifffile.tifffile.TiffPage
        Tiff page recorded from a camera in Bluelake.
    """

    def __init__(self, page, align):
        self._src = page
        self._description = ImageDescription(page)
        self._align = align

    def _align_image(self):
        """ reconstruct image using alignment matrices from Bluelake; return aligned image as a NumPy array"""

        if not self._description:
            warnings.warn("File does not contain metadata. Only raw data is available")
            return self.raw_data

        try:
            align_mats = [self._description.alignment_matrix(color) for color in ("red", "blue")]
        except KeyError:
            warnings.warn("File does not contain alignment matrices. Only raw data is available")
            return self.raw_data

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
        return self._align_image() if (self.is_rgb and self._align) else self._src.asarray()

    @property
    def raw_data(self):
        return self._src.asarray()

    @property
    def bit_depth(self):
        bit_depth = self._src.tags["BitsPerSample"].value
        if self.is_rgb:  # (int r, int g, int b)
            return bit_depth[0]
        else:  # int
            return bit_depth

    @property
    def is_rgb(self):
        return self._src.tags["SamplesPerPixel"].value == 3

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
        timestamp_string = re.search(r"^(\d+):\d+$", self._src.tags["DateTime"].value)
        return np.int64(timestamp_string.group(1)) if timestamp_string else None

    @property
    def stop(self):
        timestamp_string = re.search(r"^\d+:(\d+)$", self._src.tags["DateTime"].value)
        return np.int64(timestamp_string.group(1)) if timestamp_string else None


class TiffStack:
    """TIFF images exported from Bluelake

    Parameters
    ----------
    tiff_file : tifffile.TiffFile
        TIFF file recorded from a camera in Bluelake.
    """

    def __init__(self, tiff_file, align):
        self._src = tiff_file
        self._align = align

    def get_frame(self, frame):
        return TiffFrame(self._src.pages[frame], align=self._align)

    @staticmethod
    def from_file(image_file, align):
        return TiffStack(tifffile.TiffFile(image_file), align=align)

    @property
    def num_frames(self):
        return len(self._src.pages)


class ImageDescription:
    def __init__(self, src):
        try:
            self.json = json.loads(src.description)
        except json.decoder.JSONDecodeError:
            self.json = {}
            self._cmap = {}
            return

        self._cmap = {"red": 0, "green": 1, "blue": 2}

        # update format if necessary
        if "Alignment red channel" in self.json:
            for color, j in self._cmap.items():
                self.json[f"Channel {j} alignment"] = self.json.pop(f"Alignment {color} channel")
                self.json[f"Channel {j} detection wavelength (nm)"] = "N/A"

    def __bool__(self):
        return bool(self.json)

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
        if self:
            for j in range(3):
                out[f"Applied channel {j} alignment"] = out.pop(f"Channel {j} alignment")
        return json.dumps(out, indent=4)
