from .detail.confocal import BaseScan, _deprecate_basescan_plot_args
from .detail.plotting import get_axes


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

    def _get_plot_data(self, channel):
        """Get photon count :class:`~lumicks.pylake.channel.Slice` for requested channel."""
        return getattr(self, f"{channel}_photon_count")

    @_deprecate_basescan_plot_args
    def plot(self, channel="rgb", *, axes=None, show_title=True, **kwargs):
        """Plot photon counts for the selected channel(s).

        Parameters
        ----------
        channel : {"red", "green", "blue", "rgb"}, optional
            Color channel to plot.
        axes : matplotlib.axes.Axes, optional
            If supplied, the axes instance in which to plot.
        show_title : bool, optional
            Controls display of auto-generated plot title
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`

        Returns
        -------
        List[matplotlib.lines.Line2D]
            A list of lines representing the plotted data.
        """
        channels = ["red", "green", "blue"] if channel == "rgb" else [channel]
        axes = get_axes(axes=axes)
        for channel in channels:
            count = self._get_plot_data(channel)
            time = (count.timestamps - count.timestamps[0]) * 1e-9
            axes.plot(time, count.data, **{"color": channel, "label": channel, **kwargs})

        axes.set_xlabel("time (s)")
        axes.set_ylabel(r"photon count")
        if show_title:
            axes.set_title(self.name)

        return axes.get_lines()
