import numpy as np
import os
import re

from .channel import Slice, TimeSeries


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
        timestamp_string = re.search(r'^(\d+):\d+$', self._src.tags['DateTime'].value)
        return np.int64(timestamp_string.group(1)) if timestamp_string else None

    @property
    def stop(self):
        timestamp_string = re.search(r'^\d+:(\d+)$', self._src.tags['DateTime'].value)
        return np.int64(timestamp_string.group(1)) if timestamp_string else None


class TiffStack:
    """TIFF images exported from Bluelake

    Parameters
    ----------
    tiff_file : tifffile.TiffFile
        TIFF file recorded from a camera in Bluelake.
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


class CorrelatedStack:
    """CorrelatedStack obtained with Bluelake.

    Parameters
    ----------
    image_name : str
        Filename for the image stack. Typically a TIFF file recorded from a camera in Bluelake.

    Examples
    --------
    ::

        Loading a stack.
            from lumicks import pylake
            stack = pylake.CorrelatedStack.from_file("example.tiff")

        Making a plot where force is correlated to images in the stack.
            file = pylake.File("example.h5")
            stack.plot_correlated(file.force1x)

        Determine the force trace averaged over frame 2...9.
            file.force1x.downsampled_over(stack[2:10].timestamps)
    """
    def __init__(self, image_name):
        self.src = TiffStack.from_file(image_name)
        self.name = os.path.splitext(os.path.basename(image_name))[0]
        self.start_idx = 0
        self.stop_idx = self.src.num_frames

    def __getitem__(self, item):
        """All indexing is in frames"""
        if isinstance(item, slice):
            if item.step is not None:
                raise IndexError("Slice steps are not supported")

            start, stop, _ = item.indices(self.num_frames)
            return CorrelatedStack.from_data(self.src, self.name, self.start_idx + start, self.start_idx + stop)
        else:
            item = self.start_idx + item if item >= 0 else self.stop_idx + item
            if item >= self.stop_idx or item < self.start_idx:
                raise IndexError("Index out of bounds")
            return CorrelatedStack.from_data(self.src, self.name, item, item + 1)

    def __iter__(self):
        idx = 0
        while idx < self.num_frames:
            yield self._get_frame(idx)
            idx += 1

    @classmethod
    def from_data(cls, data, name=None, start_idx=0, stop_idx=None):
        """Construct CorrelatedStack from image stack object

        Parameters
        ----------
        data : TiffStack
            TiffStack object.
        name : str
            Plot label of the correlated stack
        start_idx : int
            Index at the first frame.
        stop_idx: int
            Index beyond the last frame.
        """
        new_correlated_stack = cls.__new__(cls)
        new_correlated_stack.src = data
        new_correlated_stack.name = name
        new_correlated_stack.start_idx = start_idx
        new_correlated_stack.stop_idx = (new_correlated_stack.src.num_frames if stop_idx is None else stop_idx)
        return new_correlated_stack

    def plot(self, frame=0, **kwargs):
        import matplotlib.pyplot as plt

        default_kwargs = dict(
            cmap='gray'
        )

        image = self.get_frame(frame).data
        plt.imshow(image, **{**default_kwargs, **kwargs})

        if self.num_frames == 1:
            plt.title(self.name)
        else:
            plt.title(f"{self.name} [frame {frame}/{self.num_frames}]")

    def _get_frame(self, frame=0):
        if frame >= self.num_frames or frame < 0:
            raise IndexError("Frame index out of range")
        return self.src.get_frame(self.start_idx + frame)

    def plot_correlated(self, channel_slice, frame=0, reduce=np.mean):
        """Downsample channel on a frame by frame basis. The downsampling function (e.g. np.mean) is evaluated for the
        time between a start and end time of a frame.

        Parameters
        ----------
        channel_slice : pylake.channel.Slice
            Data slice that we with to downsample.
        frame : int
            Frame to show.
        reduce : callable
            The `numpy` function which is going to reduce multiple samples into one.
            The default is `np.mean`, but `np.sum` could also be appropriate for some
            cases, e.g. photon counts.


        Examples
        --------
        ::

            from lumicks import pylake

            file = pylake.File("example.h5")
            stack = pylake.CorrelatedStack("example.tiff")
            stack.plot_correlated(file.force1x, frame=5)
        """
        import matplotlib.pyplot as plt

        downsampled = channel_slice.downsampled_over(self.timestamps, where='left')
        fetched_frame = self._get_frame(self.start_idx + frame)
        aspect_ratio = fetched_frame.data.shape[0] / np.max([fetched_frame.data.shape])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plt.figaspect(aspect_ratio/2))
        t0 = downsampled.timestamps[0]
        t, y = (downsampled.timestamps - t0)/1e9, downsampled.data
        ax1.step(t, y, where='pre')
        ax2.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        image_object = ax2.imshow(fetched_frame.data, cmap='gray')

        # Make sure the y-axis limits stay fixed when we add our little indicator rectangle
        y1, y2 = ax1.get_ylim()
        ax1.set_ylim(y1, y2)

        def update_position(new_frame):
            return ax1.fill_between((np.array([new_frame.start, new_frame.stop]) - t0)/1e9, y1, y2, alpha=0.7, color='r')

        poly = update_position(fetched_frame)

        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel(downsampled.labels['y'])
        ax1.set_title(downsampled.labels['title'])
        ax1.set_xlim([np.min(t), np.max(t)])

        def select_frame(event):
            nonlocal poly

            if not event.canvas.widgetlock.locked() and event.inaxes == ax1:
                time = event.xdata * 1e9 + t0
                for img_idx in np.arange(self.start_idx, self.stop_idx):
                    current_frame = self._get_frame(self.start_idx + img_idx)

                    if current_frame.start < time < current_frame.stop:
                        poly.remove()
                        image_object.set_data(current_frame.data)
                        poly = update_position(current_frame)
                        return

        fig.canvas.mpl_connect('button_press_event', select_frame)

    @property
    def num_frames(self):
        return self.stop_idx - self.start_idx

    @property
    def raw(self):
        if self.num_frames > 1:
            return [self._get_frame(idx) for idx in range(self.num_frames)]
        else:
            return self._get_frame(0)

    @property
    def start(self):
        return self._get_frame(0).start

    @property
    def stop(self):
        return self._get_frame(self.num_frames - 1).stop

    @property
    def timestamps(self):
        return [(self._get_frame(idx).start, self._get_frame(idx).stop) for idx in range(self.num_frames)]
