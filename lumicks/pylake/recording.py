import numpy as np
import os
import re

from .channel import Slice, Continuous, TimeSeries

class TiffFrame:
    """Thin wrapper around a TIFF frame stack. For camera videos timestamps are stored in the DateTime tag in
    the format start:end.

    Parameters
    ----------
    page : tifffile.tifffile.TiffPage
        Tiff page recorded from a camera in Bluelake.
    """
    def __init__(self, page):
        self._src = page

    @property
    def data(self):
        return self._src.asarray()

    @property
    def start(self):
        timestamp_string = re.search('^(\d+):\d+$', self._src.tags['DateTime'].value)
        return np.int64(timestamp_string.group(1)) if timestamp_string else None

    @property
    def stop(self):
        timestamp_string = re.search('^\d+:(\d+)$', self._src.tags['DateTime'].value)
        return np.int64(timestamp_string.group(1)) if timestamp_string else None


class TiffStack:
    """TIFF images exported from Bluelake

        Parameters
        ----------
        tiff_file : tifffile.TiffFile
            Filename pointing to a TIFF file recorded from a camera in Bluelake.
        """
    def __init__(self, tiff_file):
        self._src = tiff_file

    def get_frame(self, frame):
        return TiffFrame(self._src.pages[frame])

    @staticmethod
    def from_file(image_file):
        import tifffile
        return TiffStack(tifffile.TiffFile(image_file))

    @property
    def num_frames(self):
        return len(self._src.pages)


class Recording:
    """Recording obtained with Bluelake

        Parameters
        ----------
        data : TiffStack
            TiffStack object.
        name : str
            Plot label of the recording
        start_idx : int
            Index at the first frame.
        stop_idx: int
            Index beyond the last frame.
        """
    def __init__(self, data, name=None, start_idx=0, stop_idx=None):
        self.src = data
        self.name = name
        self.start_idx = start_idx
        self.stop_idx = (self.src.num_frames if stop_idx is None else stop_idx)

    def __getitem__(self, item):
        """All indexing is in frames"""
        if isinstance(item, slice):
            if item.step is not None:
                raise IndexError("Slice steps are not supported")

            start, stop, _ = item.indices(self.num_frames)
            return Recording(self.src, self.name, self.start_idx + start, self.start_idx + stop)
        else:
            item = self.start_idx + item if item >= 0 else self.stop_idx + item
            if item >= self.stop_idx or item < self.start_idx:
                raise IndexError("Index out of bounds")
            return Recording(self.src, self.name, item, item + 1)

    @staticmethod
    def from_file(image_name):
        """Construct recording from file

        Parameters
        ----------
        image_name : str
            Filename for the image stack.
        """
        return Recording(TiffStack.from_file(image_name), os.path.splitext(os.path.basename(image_name))[0])

    def plot(self,  **kwargs):
        import matplotlib.pyplot as plt

        default_kwargs = dict(
            cmap='gray'
        )

        frame = np.clip(kwargs.pop("frame", 0), 0, self.num_frames)
        image = self.get_frame(frame).data
        plt.imshow(image, **{**default_kwargs, **kwargs})

        if self.num_frames == 1:
            plt.title(self.name)
        else:
            plt.title(f"{self.name} [frame {frame}/{self.num_frames}]")

    def get_frame(self, frame=0):
        if frame >= self.num_frames or frame < 0:
            raise IndexError("Frame index out of range")
        return self.src.get_frame(self.start_idx + frame)

    def downsample_channel(self, channel_slice, reduce=np.mean):
        """Downsample channel on a frame by frame basis. The downsampling function (e.g. np.mean) is evaluated for the
        time between a start and end time of a frame. A list is returned that contains the data corresponding to each
        frame"

        Parameters
        ----------
        channel_slice : pylake.channel.Slice
            Data slice that we with to downsample.
        reduce : callable
            The `numpy` function which is going to reduce multiple samples into one.
            The default is `np.mean`, but `np.sum` could also be appropriate for some
            cases, e.g. photon counts.
        """
        t = np.zeros(self.num_frames)
        d = np.zeros(self.num_frames)
        for i, img_idx in enumerate(np.arange(self.start_idx, self.stop_idx)):
            start, stop = (self.src.get_frame(img_idx).start, self.src.get_frame(img_idx).stop)
            subset = channel_slice[start:stop]
            t[i] = (start + stop) // 2
            d[i] = reduce(subset.data)

        return Slice(TimeSeries(d, t))

    @property
    def time_slice(self):
        return slice(self.src.get_frame(self.start_idx).start, self.src.get_frame(self.stop_idx-1).stop)

    @property
    def num_frames(self):
        return self.stop_idx - self.start_idx
