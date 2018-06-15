import numpy as np

from .detail.timeindex import to_timestamp


class Slice:
    """A lazily evaluated slice of a timeline/HDF5 channel

    Users will only ever get these as a result of slicing a timeline/HDF5
    channel or slicing another slice (via this classes `__getitem__`), i.e.
    the `__init__` method will never be invoked by users.

    Parameters
    ----------
    data_source : Any
        A slice data source. Can be `Continuous`, `TimeSeries` or any other source
        which conforms to the same interface.
    labels : Dict[str, str]
        Plot labels: "x", "y", "title".
    """
    def __init__(self, data_source, labels=None):
        self._src = data_source
        self.labels = labels or {}

    def __len__(self):
        return len(self._src)

    def __getitem__(self, item):
        """All indexing is in timestamp units (ns)"""
        if not isinstance(item, slice):
            raise IndexError("Scalar indexing is not supported, only slicing")
        if item.step is not None:
            raise IndexError("Slice steps are not supported")

        if len(self) == 0:
            return self

        src_start = self._src.start
        src_stop = self._src.stop

        start, stop = item.start, item.stop
        if start is None:
            start = src_start
        if stop is None:
            stop = src_stop
        start, stop = (to_timestamp(v, src_start, src_stop) for v in (start, stop))

        return self.__class__(self._src.slice(start, stop), self.labels)

    @property
    def data(self):
        """The primary values of this channel slice"""
        return self._src.data

    @property
    def timestamps(self):
        """Absolute timestamps (since epoch) which correspond to the channel data"""
        return self._src.timestamps

    def plot(self, **kwargs):
        """A simple line plot to visualize the data over time"""
        import matplotlib.pyplot as plt

        seconds = (self.timestamps - self.timestamps[0]) / 1e9
        plt.plot(seconds, self.data, **kwargs)
        plt.xlabel(self.labels.get("x", "Time") + " (s)")
        plt.ylabel(self.labels.get("y", "y"))
        plt.title(self.labels.get("title", "title"))


class Continuous:
    """A source of continuous data for a timeline slice

    Parameters
    ----------
    data : array_like
        Anything that's convertible to an `np.ndarray`.
    start : int
        Timestamp of the first data point.
    dt : int
        Delta between two timestamps. Constant for the entire data range.
    """
    def __init__(self, data, start, dt):
        self._src_data = data
        self._cached_data = None
        self.start = start
        self.stop = start + len(data) * dt
        self.dt = dt

    def __len__(self):
        return len(self._src_data)

    @property
    def data(self):
        if self._cached_data is None:
            self._cached_data = np.asarray(self._src_data)
        return self._cached_data

    @property
    def timestamps(self):
        return np.arange(self.start, self.stop, self.dt)

    def slice(self, start, stop):
        # TODO: should be lazily evaluated
        idx = np.logical_and(start <= self.timestamps, self.timestamps < stop)
        return self.__class__(self.data[idx], max(start, self.start), self.dt)


class Timeseries:
    """A source of time series data for a timeline slice

    Parameters
    ----------
    data : array_like
        Anything that's convertible to an `np.ndarray`.
    timestamps : array_like
        An array of integer timestamps.
    """
    def __init__(self, data, timestamps):
        assert len(data) == len(timestamps)
        # TODO: should be lazily evaluated
        self.data = np.asarray(data)
        self.timestamps = np.asarray(timestamps)

    def __len__(self):
        return len(self.data)

    @property
    def start(self):
        return self.timestamps[0]

    @property
    def stop(self):
        return self.timestamps[-1] + 1

    def slice(self, start, stop):
        idx = np.logical_and(start <= self.timestamps, self.timestamps < stop)
        return self.__class__(self.data[idx], self.timestamps[idx])


class Empty:
    """A lightweight source of no data

    Both `Continuous` and `Timeseries` can be empty, but this is a lighter
    class which can be returned an empty slice from properties.
    """
    def __len__(self):
        return 0

    @property
    def data(self):
        return np.empty(0)

    @property
    def timestamps(self):
        return np.empty(0)


empty_slice = Slice(Empty())


def is_continuous_channel(dset):
    """Continuous channels consist of simple (non-compound) types"""
    return dset.dtype.fields is None


def make_continuous_channel(dset, y_label="y"):
    """Generate timestamps based on start/stop time and the sample rate"""
    start = dset.attrs["Start time (ns)"]
    dt = int(1e9 / dset.attrs["Sample rate (Hz)"])
    return Slice(Continuous(dset.value, start, dt),
                 labels={"title": dset.name.strip("/"), "y": y_label})


def make_timeseries_channel(dset, y_label="y"):
    """Just real all the data"""
    return Slice(Timeseries(dset["Value"], dset["Timestamp"]),
                 labels={"title": dset.name.strip("/"), "y": y_label})
