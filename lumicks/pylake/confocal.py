import json

from .detail.mixin import PhotonCounts
from .detail.mixin import ExcitationLaserPower


"""Axis label used for plotting"""
axis_label = ("x", "y", "z")


class BaseScan(PhotonCounts, ExcitationLaserPower):
    """Base class for confocal scans

    Parameters
    ----------
    name : str
        confocal scan name
    file : lumicks.pylake.File
        Parent file. Contains the channel data.
    start : int
        Start point in the relevant info wave.
    stop : int
        End point in the relevant info wave.
    json : dict
        Dictionary containing metadata.
    """

    def __init__(self, name, file, start, stop, json):
        self.start = start
        self.stop = stop
        self.name = name
        self.json = json
        self.file = file
        self._cache = {}

    @classmethod
    def from_dataset(cls, h5py_dset, file):
        """
        Construct a confocal class from dataset.

        Parameters
        ----------
        h5py_dset : h5py.Dataset
            The original HDF5 dataset containing confocal scan information
        file : lumicks.pylake.File
            The parent file. Used to loop up channel data
        """
        start = h5py_dset.attrs["Start time (ns)"]
        stop = h5py_dset.attrs["Stop time (ns)"]
        name = h5py_dset.name.split("/")[-1]
        try:
            json_data = json.loads(h5py_dset[()])["value0"]
        except KeyError:
            # TODO => use class name
            raise KeyError(f"Scan '{name}' is missing metadata and cannot be loaded")

        return cls(name, file, start, stop, json_data)


class ConfocalImage():

    @property
    def pixels_per_line(self):
        return self._fast_axis_metadata["num of pixels"]     

    @property
    def _fast_axis_metadata(self):
        return self.json["scan volume"]["scan axes"][0]

    @property
    def fast_axis(self):
        return "X" if self._fast_axis_metadata["axis"] == 0 else "Y"