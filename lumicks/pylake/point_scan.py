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

    def _plot_color(self, color, **kwargs):
        import matplotlib.pyplot as plt

        count = getattr(self, f"{color}_photon_count")
        time = (count.timestamps - count.timestamps[0]) * 1e-9
        plt.plot(time, count.data, **{"color": color, "label": color, **kwargs})
        plt.xlabel("time (s)")
        plt.ylabel(r"photon count")
        plt.title(self.name)

    def plot_rgb(self, **kwargs):
        """Plot all color channels

        Parameters
        ----------
        **kwargs
            Forwarded to `~matplotlib.pyplot.plot`.
        """
        for color in ["red", "green", "blue"]:
            self._plot_color(color, **kwargs)
