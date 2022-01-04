from .detail.confocal import BaseScan


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
        """Get photon count `Slice` for requested channel."""
        return getattr(self, f"{channel}_photon_count")

    def _plot(self, channel, axes, **kwargs):
        """Plot photon counts for the selected channel(s).

        Parameters
        ----------
        channe : {'red', 'green', 'blue', 'rgb'}
            Color channel to plot
        axes : mpl.axes.Axes or None
            If supplied, the axes instance in which to plot.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`
        """
        channels = ["red", "green", "blue"] if channel == "rgb" else [channel]
        for channel in channels:
            count = self._get_plot_data(channel)
            time = (count.timestamps - count.timestamps[0]) * 1e-9
            axes.plot(time, count.data, **{"color": channel, "label": channel, **kwargs})
            axes.set_xlabel("time (s)")
            axes.set_ylabel(r"photon count")
            axes.set_title(self.name)
