import numpy as np

from .detail.timeindex import to_timestamp


class Slice:
    """A lazily evaluated* slice of a timeline channel

    *Not actually lazy, yet.

    Users will only ever get these as a result of slicing a timeline channel
    or slicing another slice (via this classes `__getitem__`).

    Attributes
    ----------
    data : array_like
        Channel data from the timeline.
    timestamps : array_like
        Timestamps in nanoseconds for each data point.
    """

    def __init__(self, data, timestamps, labels=None):
        assert len(data) == len(timestamps)
        self.data = np.asarray(data)
        self.timestamps = np.asarray(timestamps)
        self.labels = labels or {}

    def __array__(self):
        """Coerce this slice into an `np.ndarray` of data values (no timestamps)"""
        # TODO: Currently, this just returns pre-formed data array, but it should eventually
        #       do lazy evaluation by only loading timeline data here on demand.
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """All indexing is in timestamp units (ns)"""
        if len(self) == 0:
            return Slice([], [], self.labels)

        first = self.timestamps[0]
        after_last = self.timestamps[-1] + 1

        if isinstance(item, slice):
            if item.step is not None:
                raise IndexError("Slice steps are not supported")

            start, stop = item.start, item.stop
            if start is None:
                start = first
            if stop is None:
                stop = after_last
            start, stop = (to_timestamp(v, first, after_last) for v in (start, stop))

            idx = np.logical_and(start <= self.timestamps, self.timestamps < stop)
            return Slice(self.data[idx], self.timestamps[idx], self.labels)
        else:
            idx = np.argwhere(self.timestamps == to_timestamp(item, first, after_last))
            return Slice(self.data[idx], self.timestamps[idx], self.labels)

    def plot(self, **kwargs):
        import matplotlib.pyplot as plt

        seconds = (self.timestamps - self.timestamps[0]) / 1e9
        plt.plot(seconds, self.data, **kwargs)
        plt.xlabel(self.labels.get("x", "Time") + " (s)")
        plt.ylabel(self.labels.get("y", "y"))
        plt.title(self.labels.get("title", "title"))


def is_continuous_channel(dset):
    """Continuous channels consist of simple (non-compound) types"""
    return dset.dtype.fields is None


def make_continuous_channel(dset, y_label="y"):
    """Generate timestamps based on start/stop time and the sample rate"""
    start = dset.attrs["Start time (ns)"]
    stop = dset.attrs["Stop time (ns)"]
    dt = int(1e9 / dset.attrs["Sample rate (Hz)"])
    return Slice(dset, np.arange(start, stop, dt),
                 labels={"title": dset.name.strip("/"), "y": y_label})


def make_timeseries_channel(dset, y_label="y"):
    """Just real all the data"""
    return Slice(dset["Value"], dset["Timestamp"],
                 labels={"title": dset.name.strip("/"), "y": y_label})
