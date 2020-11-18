import json
import numpy as np
import warnings

from .detail.confocal import ConfocalImage
from .detail.image import reconstruct_image_sum, reconstruct_image, line_timestamps_image, seek_timestamp_next_line
from .detail.timeindex import to_timestamp


class Kymo(ConfocalImage):
    """A Kymograph exported from Bluelake

    Parameters
    ----------
    name : str
        Kymograph name
    file : lumicks.pylake.File
        Parent file. Contains the channel data.
    start : int
        Start point in the relevant info wave.
    stop : int
        End point in the relevant info wave.
    json : dict
        Dictionary containing kymograph-specific metadata.
    """

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(pixels={self.pixels_per_line})"

    def __getitem__(self, item):
        """All indexing is in timestamp units (ns)"""
        if not isinstance(item, slice):
            raise IndexError("Scalar indexing is not supported, only slicing")
        if item.step is not None:
            raise IndexError("Slice steps are not supported")

        start = self.start if item.start is None else item.start
        stop = self.stop if item.stop is None else item.stop
        start, stop = (to_timestamp(v, self.start, self.stop) for v in (start, stop))

        timestamps = self.infowave.timestamps
        line_timestamps = line_timestamps_image(timestamps, self.infowave.data, self.pixels_per_line)
        line_timestamps = np.append(line_timestamps, timestamps[-1])

        i_min = np.searchsorted(line_timestamps, start, side='left')
        i_max = np.searchsorted(line_timestamps, stop, side='left')

        if i_min >= len(line_timestamps):
            return EmptyKymo(self.name, self.file, line_timestamps[-1], line_timestamps[-1], self.json)

        if i_min >= i_max:
            return EmptyKymo(self.name, self.file, line_timestamps[i_min], line_timestamps[i_min], self.json)

        if i_max < len(line_timestamps):
            stop = line_timestamps[i_max]

        start = line_timestamps[i_min]

        return Kymo(self.name, self.file, start, stop, self.json)

    def _fix_incorrect_start(self):#, timeline_start, timeline_dt):
        """ Resolve error when confocal scan starts before the timeline information.
            For kymographs this is recoverable by omitting the first line. """
        self.start = seek_timestamp_next_line(self.infowave[self.start:])
        self._cache = {}
        warnings.warn("Start of the kymograph was truncated. Omitting the truncated first line.",
                        RuntimeWarning)

    def _to_spatial(self, data):
        """Spatial data as rows, time as columns"""
        return data.T

    @property
    def _shape(self):
        return (self.pixels_per_line, )

    @property
    def line_time_seconds(self):
        """Line time in seconds"""
        if self.timestamps.shape[1] > 1:
            return (self.timestamps[0, 1] - self.timestamps[0, 0]) / 1e9
        else:
            raise RuntimeError("Line time is not defined for kymograph with only a single line")

    def _plot(self, image, **kwargs):
        import matplotlib.pyplot as plt

        width_um = self._ordered_axes()[0]["scan width (um)"]
        ts = self.timestamps
        duration = (ts[0, -1] - ts[0, 0]) / 1e9
        linetime = (ts[0, 1] - ts[0, 0]) / 1e9

        default_kwargs = dict(
            # With origin set to upper (default) bounds should be given as (0, n, n, 0)
            # pixel center aligned with mean time per line
            extent=[-0.5 * linetime, duration + 0.5 * linetime, width_um, 0],
            aspect=(image.shape[0] / image.shape[1]) * (duration / width_um)
        )

        plt.imshow(image, **{**default_kwargs, **kwargs})
        plt.xlabel("time (s)")
        plt.ylabel(r"position ($\mu$m)")
        plt.title(self.name)

    def _downsample_channel(self, n, xy, reduce=np.mean):
        force = self.file._get_force(n, xy)
        # downsample exactly over the scanline time range
        min_times = self.timestamps[0, :].astype(np.int64)
        max_times = self.timestamps[-1, :].astype(np.int64)
        time_ranges = [(mini, maxi) for mini, maxi in zip(min_times, max_times)]
        return force.downsampled_over(time_ranges, reduce=reduce, where="center")

    def plot_with_force(self, force_channel, color_channel, aspect_ratio=0.25,
                        reduce=np.mean, **kwargs):
        """Plot kymo with force channel downsampled over scan lines

        Parameters
        ----------
        force_channel: str
            name of force channel to downsample and plot
        color_channel: str
            color channel of kymo to plot ('red', 'green', 'blue', 'rgb')
        aspect_ratio: float
            aspect ratio of the axes (i.e. ratio of y-unit to x-unit)
        reduce : callable
            The `numpy` function which is going to reduce multiple samples into one.
            Forwarded to :func:`Slice.downsampled_over`
        **kwargs
            Forwarded to :func:`Slice.plot`.
        """
        def set_aspect_ratio(axis, ar):
            """This function forces a specific aspect ratio, can be useful when aligning figures"""
            axis.set_aspect(ar * np.abs(np.diff(axis.get_xlim())[0] / np.diff(axis.get_ylim()))[0])

        import matplotlib.pyplot as plt
        _, (ax1, ax2) = plt.subplots(2, 1)
        
        # plot kymo
        plt.sca(ax1)
        getattr(self, f"plot_{color_channel}")()
        ax1.set_xlabel(None)
        
        # plot force channel
        plt.sca(ax2)
        force = self._downsample_channel(force_channel[-2], force_channel[-1], reduce=reduce)
        force.plot(**kwargs)
        ax2.set_xlim(ax1.get_xlim())
        
        set_aspect_ratio(ax1, aspect_ratio)
        set_aspect_ratio(ax2, aspect_ratio)


class EmptyKymo(Kymo):
    def plot_rgb(self):
        raise RuntimeError("Cannot plot empty kymograph")

    def _plot(self, image, **kwargs):
        raise RuntimeError("Cannot plot empty kymograph")

    def _image(self, color):
        return np.empty((self.pixels_per_line, 0))
