import json

from .detail.mixin import PhotonCounts
from .detail.mixin import ExcitationLaserPower


class PointScan(PhotonCounts, ExcitationLaserPower):
    """A point scan exported from Bluelake

    Parameters
    ----------
    h5py_dset : h5py.Dataset
        The original HDF5 dataset containing the point scan
    file : lumicks.pylake.File
        The parent file. Used to look up channel data.
    """
    def __init__(self, name, file, start, stop, json):
        self.start = start
        self.stop = stop
        self.name = name
        self.json = json
        self.file = file

    def _get_photon_count(self, name):
        return getattr(self.file, f"{name}_photon_count".lower())[self.start:self.stop]

    @property
    def has_fluorescence(self) -> bool:
        return self.json["fluorescence"]

    @property
    def has_force(self) -> bool:
        return self.json["force"]

    def _plot_color(self, color, **kwargs):
        import matplotlib.pyplot as plt

        count = getattr(self, f"{color}_photon_count")
        time = (count.timestamps - count.timestamps[0]) * 1e-9
        plt.plot(time, count.data, **{"color": color, "label": color, **kwargs})
        plt.xlabel("time (s)")
        plt.ylabel(r"photon count")
        plt.title(self.name)

    def plot_red(self, **kwargs):
        """Plot the red photon channel

        Parameters
        ----------
        **kwargs
            Forwarded to `~matplotlib.pyplot.plot`.
        """
        self._plot_color("red", **kwargs)

    def plot_green(self, **kwargs):
        """Plot the red photon channel

        Parameters
        ----------
        **kwargs
            Forwarded to `~matplotlib.pyplot.plot`.
        """
        self._plot_color("green", **kwargs)

    def plot_blue(self, **kwargs):
        """Plot the red photon channel

        Parameters
        ----------
        **kwargs
            Forwarded to `~matplotlib.pyplot.plot`.
        """
        self._plot_color("blue", **kwargs)

    def plot_rgb(self, **kwargs):
        """Plot all color channels

        Parameters
        ----------
        **kwargs
            Forwarded to `~matplotlib.pyplot.plot`.
        """
        for color in ["red", "green", "blue"]:
            self._plot_color(color, **kwargs)

    @classmethod
    def from_dataset(cls, h5py_dset, file):
        """Construct PointScan class from dataset.

        Parameters
        ----------
        h5py_dset : h5py.Dataset
            The original HDF5 dataset containing kymo information
        file : lumicks.pylake.File
            The parent file. Used to loop up channel data
        """
        start = h5py_dset.attrs["Start time (ns)"]
        stop = h5py_dset.attrs["Stop time (ns)"]
        name = h5py_dset.name.split("/")[-1]
        json_data = json.loads(h5py_dset[()])["value0"]
        return cls(name, file, start, stop, json_data)
