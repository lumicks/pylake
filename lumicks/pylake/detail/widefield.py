import numpy as np
import re
import json
import cv2
import tifffile
import warnings


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
        self._align = align

    def _align_image(self):
        """ reconstruct image using alignment matrices from Bluelake; return aligned image as a NumPy array"""

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

        try:
            description = json.loads(self._src.description)
        except json.decoder.JSONDecodeError:
            warnings.warn("File does not contain metadata. Only raw data is available")
            return self.raw_data

        try:
            align_mats = [
                np.array(description[f"Alignment {color} channel"]).reshape((2, 3))
                for color in ("red", "blue")
            ]
            align_roi = np.array(description["Alignment region of interest (x, y, width, height)"])[
                :2
            ]
            roi = np.array(description["Region of interest (x, y, width, height)"])[:2]
        except KeyError:
            warnings.warn("File does not contain alignment matrices. Only raw data is available")
            return self.raw_data

        x_offset, y_offset = align_roi - roi
        if not (x_offset == 0 and y_offset == 0):
            align_mats = [correct_alignment_offset(mat, x_offset, y_offset) for mat in align_mats]

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
