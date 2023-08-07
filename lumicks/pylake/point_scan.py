from copy import copy

from .detail.confocal import BaseScan
from .detail.plotting import get_axes
from .detail.timeindex import to_timestamp


class PointScan(BaseScan):
    """A confocal point scan exported from Bluelake

    Parameters
    ----------
    name : str
        point scan name
    file : lumicks.pylake.File
        Parent file. Contains the channel data.
    start : int
        Start point in the relevant info wave.
    stop : int
        End point in the relevant info wave.
    json : dict
        Dictionary containing scan-specific metadata.
    """

    def __getitem__(self, item):
        """All indexing is in timestamp units (ns)"""
        if not isinstance(item, slice):
            raise IndexError("Scalar indexing is not supported, only slicing")
        if item.step is not None:
            raise IndexError("Slice steps are not supported")

        start = self.start if item.start is None else item.start
        stop = self.stop if item.stop is None else item.stop
        new_scan = copy(self)
        new_scan.start, new_scan.stop = (
            to_timestamp(v, self.start, self.stop) for v in (start, stop)
        )
        return new_scan

    def _get_plot_data(self, channel):
        """Get photon count :class:`~lumicks.pylake.channel.Slice` for requested channel."""
        return getattr(self, f"{channel}_photon_count")

    def plot(self, channel="rgb", *, axes=None, show_title=True, show_axes=True, **kwargs):
        """Plot photon counts for the selected channel(s).

        Parameters
        ----------
        channel : {"red", "green", "blue", "rgb"}, optional
            Color channel to plot.
        axes : matplotlib.axes.Axes, optional
            If supplied, the axes instance in which to plot.
        show_title : bool, optional
            Controls display of auto-generated plot title
        show_axes : bool, optional
            Setting show_axes to False hides the axes.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`

        Returns
        -------
        List[matplotlib.lines.Line2D]
            A list of lines representing the plotted data.
        """
        channels = ["red", "green", "blue"] if channel == "rgb" else [channel]
        axes = get_axes(axes=axes)

        if show_axes is False:
            axes.set_axis_off()

        for channel in channels:
            count = self._get_plot_data(channel)
            time = (count.timestamps - count.timestamps[0]) * 1e-9
            axes.plot(time, count.data, **{"color": channel, "label": channel, **kwargs})

        axes.set_xlabel("time (s)")
        axes.set_ylabel(r"photon count")
        if show_title:
            axes.set_title(self.name)

        return axes.get_lines()
