import numpy as np
from copy import copy
from .channel import Slice, TimeSeries
from .detail.mixin import DownsampledFD


class FDCurve(DownsampledFD):
    """An FD curve exported from Bluelake

    By default, the primary force and distance channels are `downsampled_force2`
    and `distance1`. Alternatives can be selected using `FDCurve.with_channels()`.
    Note that it does not modify the FD curve in place but returns a copy.

    Attributes
    ----------
    file : lumicks.pylake.File
        The parent file. Used to look up channel data.
    start, stop : int
        The time range (ns) of this FD curve within the file.
    name : str
        The name of this FD curve as it appeared in the timeline.
    """
    def __init__(self, file, start, stop, name, force="2", distance="1"):
        self.file = file
        self.start = start
        self.stop = stop
        self.name = name
        self._primary_force_channel = force
        self._primary_distance_channel = distance
        self._force_cache = None
        self._distance_cache = None

    @classmethod
    def from_dataset(cls, h5py_dset, file, **kwargs):
        return cls(file=file, start=h5py_dset.attrs["Start time (ns)"],
                   stop=h5py_dset.attrs["Stop time (ns)"], name=h5py_dset.name.split("/")[-1],
                   **kwargs)

    def __copy__(self):
        """Custom implementation of copy because the caches need to be cleared"""
        cls = self.__class__
        new_copy = cls.__new__(cls)
        new_copy.__dict__.update(self.__dict__)
        new_copy._force_cache = None
        new_copy._distance_cache = None
        return new_copy

    def __sub__(self, baseline):
        """Subtract FD curve `baseline` from `self`

        The resulting FD curve will be clipped to the distance range of `baseline`.
        The `baseline` force will be interpolated (using `scipy.interpolate.interp1d()`)
        onto the distance points of `self`.
        """
        from scipy.interpolate import interp1d

        baseline_distance = baseline.d.data
        baseline_force = baseline.f.data

        clipped_idx = np.logical_and(np.min(baseline_distance) <= self.d.data,
                                     self.d.data <= np.max(baseline_distance))
        distance = self.d.data[clipped_idx]
        force = self.f.data[clipped_idx]
        timestamps = self.f.timestamps[clipped_idx]

        interpolated_baseline_force = interp1d(baseline_distance, baseline_force)
        new_force = force - interpolated_baseline_force(distance)

        new_fd = copy(self)
        new_fd._force_cache = Slice(TimeSeries(new_force, timestamps), self.f.labels)
        new_fd._distance_cache = Slice(TimeSeries(distance, timestamps), self.d.labels)
        return new_fd

    def __getitem__(self, item):
        new_curve = self.__copy__()
        new_curve._force_cache = self.f[item]
        new_curve._distance_cache = self.d[item]
        new_curve.start = new_curve.f._src.start
        new_curve.stop = new_curve.f._src.stop
        return new_curve

    def _get_downsampled_force(self, n, xy):
        return getattr(self.file, f"downsampled_force{n}{xy}")[self.start:self.stop]

    def _get_distance(self, n):
        return getattr(self.file, f"distance{n}")[self.start:self.stop]

    @property
    def f(self):
        """The primary force channel associated with this FD curve"""
        if self._force_cache is None:
            self._force_cache = getattr(self, f"downsampled_force{self._primary_force_channel}")
        return self._force_cache

    @property
    def d(self):
        """The primary distance channel associated with this FD curve"""
        if self._distance_cache is None:
            self._distance_cache = getattr(self, f"distance{self._primary_distance_channel}")
        return self._distance_cache

    def with_channels(self, force, distance):
        """Return a copy of this FD curve with difference primary force and distance channels"""
        new_fd = copy(self)
        new_fd._primary_force_channel = force
        new_fd._primary_distance_channel = distance
        return new_fd

    def plot_scatter(self, **kwargs):
        """Plot the FD curve points

        Parameters
        ----------
        **kwargs
            Forwarded to `~matplotlib.pyplot.scatter`.
        """
        import matplotlib.pyplot as plt

        plt.scatter(self.d.data, self.f.data, **{"s": 8, **kwargs})
        plt.xlabel(self.d.labels.get("y", "distance"))
        plt.ylabel(self.f.labels.get("y", "force"))
        plt.title(self.name)
