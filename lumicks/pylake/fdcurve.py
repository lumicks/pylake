from .detail.mixin import DownsampledFD


class FDCurve(DownsampledFD):
    """An FD curve exported from Bluelake

    By default, the primary force and distance channels are `downsampled_force2`
    and `distance1`. Alternatives can be selected using `FDCurve.with_channels()`.
    Note that it does not modify the FD curve in place but returns a copy.

    Parameters
    ----------
    h5py_dset : h5py.Dataset
        The original HDF5 dataset containing FD curve information
    file : lumicks.pylake.File
        The parent file. Used to loop up channel data.
    """
    def __init__(self, h5py_dset, file, force="2", distance="1"):
        self.dset = h5py_dset
        self.start = h5py_dset.attrs["Start time (ns)"]
        self.stop = h5py_dset.attrs["Stop time (ns)"]
        self.name = h5py_dset.name.split("/")[-1]
        self.file = file
        self._primary_force_channel = force
        self._primary_distance_channel = distance

    def _get_downsampled_force(self, n, xy):
        return getattr(self.file, f"downsampled_force{n}{xy}")[self.start:self.stop]

    def _get_distance(self, n):
        return getattr(self.file, f"distance{n}")[self.start:self.stop]

    @property
    def f(self):
        """The primary force channel associated with this FD curve"""
        return getattr(self, f"downsampled_force{self._primary_force_channel}")

    @property
    def d(self):
        """The primary distance channel associated with this FD curve"""
        return getattr(self, f"distance{self._primary_distance_channel}")

    def with_channels(self, force, distance):
        """Return a copy of this FD curve with difference primary force and distance channels"""
        return FDCurve(self.dset, self.file, force, distance)

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
