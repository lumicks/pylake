import numpy as np

from .detail.utilities import downsample
from .detail.timeindex import to_timestamp
from .calibration import ForceCalibration
from .nb_widgets.range_selector import SliceRangeSelectorWidget


class Slice:
    """A lazily evaluated slice of a timeline/HDF5 channel

    Users will only ever get these as a result of slicing a timeline/HDF5
    channel or slicing another slice (via this class' `__getitem__`), i.e.
    the `__init__` method will never be invoked by users.

    Parameters
    ----------
    data_source : Any
        A slice data source. Can be `Continuous`, `TimeSeries`, 'TimeTags',
        or any other source which conforms to the same interface.
    labels : Dict[str, str]
        Plot labels: "x", "y", "title".
    calibration: ForceCalibration
    """

    def __init__(self, data_source, labels=None, calibration=None):
        self._src = data_source
        self.labels = labels or {}
        self._calibration = calibration

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

        start = src_start if item.start is None else item.start
        stop = src_stop if item.stop is None else item.stop
        start, stop = (to_timestamp(v, src_start, src_stop) for v in (start, stop))

        return self._with_data_source(self._src.slice(start, stop))

    def _with_data_source(self, data_source):
        """Return a copy of this slice with a different data source, but keep other properties"""
        return self.__class__(data_source, self.labels, self._calibration)

    def _unpack_other(self, other):
        if np.isscalar(other):
            return other

        if not isinstance(other, Slice):
            raise TypeError("Trying to perform operation with incompatible types.")

        if isinstance(other._src, TimeTags):
            raise NotImplementedError("This operation is not supported for TimeTag data")

        if not np.array_equal(other.timestamps, self.timestamps):
            raise RuntimeError("Cannot perform arithmetic on slices with different timestamps.")

        return other.data

    def __add__(self, other):
        return Slice(self._src._with_data(self.data + self._unpack_other(other)))

    def __sub__(self, other):
        return Slice(self._src._with_data(self.data - self._unpack_other(other)))

    def __truediv__(self, other):
        return Slice(self._src._with_data(self.data / self._unpack_other(other)))

    def __mul__(self, other):
        return Slice(self._src._with_data(self.data * self._unpack_other(other)))

    def __pow__(self, other):
        return Slice(self._src._with_data(self.data ** self._unpack_other(other)))

    def __rtruediv__(self, other):
        return Slice(self._src._with_data(self._unpack_other(other) / self.data))

    def __rsub__(self, other):
        return Slice(self._src._with_data(self._unpack_other(other) - self.data))

    def __rpow__(self, other):
        return Slice(self._src._with_data(self._unpack_other(other) ** self.data))

    # These commute
    __rmul__ = __mul__
    __radd__ = __add__

    @property
    def start(self):
        """Starting timestamp of this time series in nanoseconds"""
        return self._src.start

    @property
    def stop(self):
        """End timestamp of this time series in nanoseconds"""
        return self._src.stop

    @property
    def data(self):
        """The primary values of this channel slice"""
        return self._src.data

    @property
    def timestamps(self):
        """Absolute timestamps (since epoch) which correspond to the channel data"""
        return self._src.timestamps

    @property
    def seconds(self):
        """Relative time (in seconds) that corresponds to the channel data"""
        return (self._src.timestamps - self._src.start) * 1e-9

    @property
    def calibration(self) -> list:
        """Calibration data slicing is deferred until calibration is requested to avoid
        slicing values that may be needed."""
        if self._calibration:
            try:
                return self._calibration.filter_calibration(self._src.start, self._src.stop)
            except IndexError:
                return []
        else:
            return []

    @property
    def sample_rate(self) -> int:
        """The data frequency for continuous data sources or `None` if it's variable"""
        try:
            return self._src.sample_rate
        except AttributeError:
            return None

    @property
    def _timesteps(self):
        return self._src._timesteps

    def downsampled_over(self, range_list, reduce=np.mean, where="center"):
        """Downsample channel data based on timestamp ranges. The downsampling function (e.g. np.mean) is evaluated for
        the time between a start and end time of each block. A list is returned that contains the data corresponding to
        each block.

        Parameters
        ----------
        range_list : list of tuples
            A list of (start, stop) tuples indicating over which ranges to apply the function.
            Start and stop have to be specified in nanoseconds.
        reduce : callable
            The `numpy` function which is going to reduce multiple samples into one.
            The default is `np.mean`, but `np.sum` could also be appropriate for some
            cases, e.g. photon counts.
        where : str
            Where to put the final time point.
            'center' time point is put at (timestamps_subset[0] + timestamps_subset[-1]) / 2, where timestamps_subset
            are the timestamps corresponding to the samples being downsampled over.
            'left' time point is put at start

        Examples
        --------
        ::

            from lumicks import pylake

            file = pylake.File("example.h5")
            stack = pylake.CorrelatedStack("example.tiff")
            file.force1x.downsampled_over(stack.timestamps)
        """
        if not isinstance(range_list, list):
            raise TypeError("Did not pass timestamps to range_list.")

        assert len(range_list[0]) == 2, "Did not pass timestamps to range_list."
        assert self._src.start < range_list[-1][1], "No overlap between range and selected channel."
        assert self._src.stop > range_list[0][0], "No overlap between range and selected channel"

        if where != "center" and where != "left":
            raise ValueError("Invalid argument for where. Valid options are center and left")

        # Only use the frames that are actually fully covered by channel data
        range_list = [
            frame_range
            for frame_range in range_list
            if frame_range[0] >= self._src.start and frame_range[1] <= self._src.stop
        ]

        t = np.zeros(len(range_list), dtype=self._src.timestamps.dtype)
        d = np.zeros(len(range_list))
        idx = 0
        for time_range in range_list:
            start, stop = time_range
            subset = self[start:stop]
            if len(subset.timestamps) > 0:
                ts = subset.timestamps
                t[idx] = (ts[0] + ts[-1]) // 2 if where == "center" else start
                d[idx] = reduce(subset.data)
                idx += 1

        return Slice(TimeSeries(d[:idx], t[:idx]), self.labels)

    def downsampled_to(self, frequency, reduce=np.mean, where="center", method="safe"):
        """Return a copy of this slice downsampled to a specified frequency

        Parameters
        ----------
        frequency : int
            The desired downsampled frequency downsampled (Hz)
        reduce : callable
            The `numpy` function which is going to reduce multiple samples into one.
            The default is `np.mean`, but `np.sum` could also be appropriate for some
            cases, e.g. photon counts.
        where : str
            Where to put the final time point.
            'center' time point is put at (timestamps_subset[0] + timestamps_subset[-1]) / 2, where timestamps_subset
            are the timestamps corresponding to the samples being downsampled over.
            'left' time point is put at start
        method : str
            How to handle target sample times that are not exact multiples of the current sample time.

            'safe'  new sample time must be an exact multiple of the current sample time,
                    else an exception is raised.
            'ceil'  rounds the sample rate up to the nearest frequency which fulfills this condition
            'force' downsample data with the target input frequency; this will result in variable
                    sample times and a variable number of sampling contributing to each target sample,
                    but must be used for variable-frequency data
        """
        if method not in ("safe", "ceil", "force"):
            raise ValueError(f"method '{method}' is not recognized")

        source_timestep = self._timesteps
        target_timestep = int(1e9 / frequency)  # Hz -> ns

        # prevent upsampling
        if np.any(target_timestep < source_timestep):
            slow = 1e9 / np.max(source_timestep)
            raise ValueError(
                f"Requested frequency ({frequency:0.1g} Hz) is faster than slowest current sampling rate "
                f"({slow:0.1g} Hz)"
            )

        if method == "force":
            pass
        else:
            # must specifically force downsampling for variable frequency
            if len(source_timestep) > 1:
                raise ValueError(
                    (
                        "This slice contains variable sampling frequencies leading to a variable timestep "
                        "and an unequal number of samples contributing to each sample. Use 'method='force' "
                        "to ignore this error and evaluate the result."
                    )
                )
            # ratio of current/new frequencies must be integer to ensure equal timesteps
            remainder = target_timestep % source_timestep
            if remainder != 0:
                if method == "ceil":
                    target_timestep -= remainder
                else:  # method == 'safe'
                    raise ValueError(
                        (
                            "The desired sample rate will not result in time steps that are an exact "
                            "multiple of the current sampling time. To round the sample rate up to the "
                            "nearest frequency which fulfills this condition please specify method='ceil'. "
                            "To force the target sample rate (leading to unequal timesteps in the returned "
                            "Slice, specify method='force'."
                        )
                    )

        t = np.arange(self._src.start, self._src.stop, target_timestep)
        new_ranges = [(t1, t2) for t1, t2 in zip(t[:-1], t[1:])]
        return self.downsampled_over(new_ranges, reduce=reduce, where=where)

    def downsampled_by(self, factor, reduce=np.mean):
        """Return a copy of this slice which is downsampled by `factor`

        Parameters
        ----------
        factor : int
            The size and sample rate of the data will be divided by this factor.
        reduce : callable
            The `numpy` function which is going to reduce multiple samples into one.
            The default is `np.mean`, but `np.sum` could also be appropriate for some
            cases, e.g. photon counts.
        """
        return self._with_data_source(self._src.downsampled_by(factor, reduce))

    def downsampled_like(self, other_slice, reduce=np.mean):
        """Downsample high frequency data analogously to a low frequency channel in the same way that Bluelake does it.

        Parameters
        ----------
        other_slice : Slice
            Timeline channel to downsample like. This should be a low frequency channel that provides the timestamps
            to downsample to.
        reduce : callable
            The `numpy` function which is going to reduce multiple samples into one.
            The default is `np.mean`, but `np.sum` could also be appropriate for some
            cases, e.g. photon counts.
        """
        assert isinstance(other_slice._src, TimeSeries), (
            "You did not pass a low frequency channel to serve as " "reference channel."
        )

        if not isinstance(self._src, Continuous):
            raise NotImplementedError(
                "Downsampled_like is only available for high frequency channels"
            )

        timestamps = other_slice.timestamps
        delta_time = np.diff(timestamps)

        assert self._src.start < timestamps[-1], "No overlap between range and selected channel."
        assert (
            self._src.stop > timestamps[0] - delta_time[0]
        ), "No overlap between range and selected channel"

        # When the frame rate changes, one frame is very long due to the delay of the camera. It should default to
        # the new frame rate.
        (change_points,) = np.nonzero(np.abs(np.diff(delta_time) > 0))
        try:
            for i in change_points:
                delta_time[i + 1] = delta_time[i + 2]
        except IndexError:
            pass

        # We copy the front delta_time to keep indices in sync with the main timeline.
        delta_time = np.hstack((delta_time[0], delta_time))

        start_idx = np.searchsorted(timestamps - delta_time[0], self._src.start)
        stop_idx = np.searchsorted(timestamps, self._src.stop)
        timestamps = timestamps[start_idx:stop_idx]
        delta_time = delta_time[start_idx:stop_idx]

        downsampled = np.array(
            [
                reduce(self[start:stop].data)
                for start, stop in zip(timestamps - delta_time, timestamps)
            ]
        )

        new_slice = self._with_data_source(TimeSeries(downsampled, timestamps))
        return new_slice, other_slice[new_slice._src.start : new_slice._src.stop]

    def plot(self, start=None, **kwargs):
        """A simple line plot to visualize the data over time

        Parameters
        ----------
        start : int64
            Origin timestamp. This can be used to plot two slices starting at different times on the same axis.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`.
        """
        import matplotlib.pyplot as plt

        start = start if start is not None else self._src.start
        plt.plot((self._src.timestamps - start) * 1e-9, self.data, **kwargs)
        plt.xlabel(self.labels.get("x", "Time") + " (s)")
        plt.ylabel(self.labels.get("y", "y"))
        plt.title(self.labels.get("title", "title"))

    def range_selector(self, show=True):
        return SliceRangeSelectorWidget(self, show=show)


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

    def _with_data(self, data):
        return self.__class__(data, self.start, self.dt)

    def __len__(self):
        return len(self._src_data)

    @staticmethod
    def from_dataset(dset, y_label="y", calibration=None):
        start = dset.attrs["Start time (ns)"]
        dt = int(1e9 / dset.attrs["Sample rate (Hz)"])
        return Slice(
            Continuous(dset, start, dt),
            labels={"title": dset.name.strip("/"), "y": y_label},
            calibration=calibration,
        )

    @property
    def data(self):
        if self._cached_data is None:
            self._cached_data = np.asarray(self._src_data)
        return self._cached_data

    @property
    def timestamps(self):
        return np.arange(self.start, self.stop, self.dt)

    @property
    def sample_rate(self):
        return int(1e9 / self.dt)

    @property
    def _timesteps(self):
        return np.asarray([self.dt], dtype=int)

    def slice(self, start, stop):
        def to_index(t):
            """Convert a timestamp into a continuous channel index (assumes t >= self.start)"""
            return (t - self.start + self.dt - 1) // self.dt

        fraction = (start - self.start) % self.dt
        start = max(start if fraction == 0 else start + self.dt - fraction, self.start)

        start_idx = to_index(start)
        stop_idx = to_index(stop)
        return self.__class__(self.data[start_idx:stop_idx], start, self.dt)

    def downsampled_by(self, factor, reduce):
        return self.__class__(
            downsample(self.data, factor, reduce),
            start=self.start + self.dt * (factor - 1) // 2,
            dt=self.dt * factor,
        )


class TimeSeries:
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

    def _with_data(self, data):
        return self.__class__(data, self.timestamps)

    @staticmethod
    def from_dataset(dset, y_label="y", calibration=None):
        return Slice(
            TimeSeries(dset["Value"], dset["Timestamp"]),
            labels={"title": dset.name.strip("/"), "y": y_label},
            calibration=calibration,
        )

    @property
    def start(self):
        if len(self.timestamps) > 0:
            return self.timestamps[0]
        else:
            raise IndexError("Start of empty time series is undefined")

    @property
    def stop(self):
        if len(self.timestamps) > 0:
            return self.timestamps[-1] + 1
        else:
            raise IndexError("End of empty time series is undefined")

    @property
    def _timesteps(self):
        return np.unique(np.diff(self.timestamps)).astype(int)

    def slice(self, start, stop):
        idx = np.logical_and(start <= self.timestamps, self.timestamps < stop)
        return self.__class__(self.data[idx], self.timestamps[idx])

    def downsampled_by(self, factor, reduce):
        raise NotImplementedError("Downsampling is currently not available for time series data")


class TimeTags:
    """A source of time tag data for a timeline slice

    Parameters
    ----------
    data : array_like
        Anything that's convertible to an `np.ndarray`
    start : int
        Timestamp of the start of the channel slice
    stop : int
        Timestamp of the end of the channel slice
    """

    def __init__(self, data, start=None, stop=None):
        self.data = np.asarray(data, dtype=np.int64)
        self.start = start if start is not None else (self.data[0] if self.data.size > 0 else 0)
        self.stop = stop if stop is not None else (self.data[-1] + 1 if self.data.size > 0 else 0)

    def __len__(self):
        return self.data.size

    def _with_data(self, data):
        raise NotImplementedError("Time tags do not currently support this operation")

    @staticmethod
    def from_dataset(dset, y_label="y"):
        return Slice(TimeTags(dset))

    @property
    def timestamps(self):
        # For time tag data, the data is the timestamps!
        return self.data

    @property
    def _timesteps(self):
        raise NotImplementedError("_timesteps is currently not available for time series data")

    def slice(self, start, stop):
        idx = np.logical_and(start <= self.data, self.data < stop)
        return self.__class__(self.data[idx], min(start, stop), max(start, stop))

    def downsampled_by(self, factor, reduce):
        raise NotImplementedError("Downsampling is not available for time tag data")


class Empty:
    """A lightweight source of no data

    Both `Continuous` and `TimeSeries` can be empty, but this is a lighter
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

    @property
    def _timesteps(self):
        raise NotImplementedError("_timesteps is currently not available for time series data")


empty_slice = Slice(Empty())


def channel_class(dset):
    """Figure out the right channel source class given an HDF5 dataset"""
    if "Kind" in dset.attrs:
        # Bluelake HDF5 files >=v2 mark channels with a "Kind" attribute:
        kind = dset.attrs["Kind"]
        if isinstance(kind, bytes):
            kind = kind.decode()

        if kind == "TimeTags":
            return TimeTags
        elif kind == "TimeSeries":
            return TimeSeries
        elif kind == "Continuous":
            return Continuous
        else:
            raise RuntimeError("Unknown channel kind " + str(kind))
    elif dset.dtype.fields is None:
        # Version 1 Bluelake HDF5 files do not have a kind field which indicates the type of data they store. These
        # older files typically contain either Continuous or TimeSeries channel data. Continuous channel data contains
        # the attribute "Sample rate (Hz)". Newer Bluelake HDF5 files also have fields which do not have the kind
        # attribute, but which are not Continuous data. Direct access to these fields is not supported as they are
        # typically accessed through dedicated API classes.
        if "Sample rate (Hz)" in dset.attrs.keys():
            return Continuous
        else:
            raise IndexError("Direct access to this field is not supported.")
    else:
        return TimeSeries
