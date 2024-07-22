from __future__ import annotations

import numbers
from typing import Union

import numpy as np
import numpy.typing as npt

from .detail.timeindex import to_timestamp
from .detail.utilities import downsample
from .nb_widgets.range_selector import SliceRangeSelectorWidget


class Slice:
    """A lazily evaluated slice of a timeline/HDF5 channel

    Users will only ever get this as a result of slicing a timeline/HDF5 channel or slicing another
    slice (via this class's :meth:`__getitem__()`), i.e. the `__init__()` method should never be
    directly invoked by users.

    Parameters
    ----------
    data_source : Any
        A slice data source. Can be `Continuous`, `TimeSeries`, `TimeTags`,
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
        if isinstance(item, np.ndarray) and (item.dtype == bool):
            return self._apply_mask(item)
        if isinstance(item, slice) and item.step is not None:
            raise IndexError("Slice steps are not supported")
        if not hasattr(item, "start") or not hasattr(item, "stop"):
            raise IndexError("Scalar indexing is not supported, only slicing")

        if len(self) == 0:
            return self

        src_start = self._src.start
        src_stop = self._src.stop

        start = src_start if item.start is None else item.start
        stop = src_stop if item.stop is None else item.stop
        start, stop = (to_timestamp(v, src_start, src_stop) for v in (start, stop))

        if not (isinstance(start, numbers.Number) and isinstance(stop, numbers.Number)):
            raise TypeError(
                "Could not evaluate [start:stop] interval. Start and stop values should be valid timestamps or time strings"
            )

        return self._with_data_source(self._src.slice(start, stop))

    def _with_data_source(self, data_source):
        """Return a copy of this slice with a different data source, but keep other properties"""
        return self.__class__(data_source, self.labels, self._calibration)

    def _apply_mask(self, mask):
        """Apply a logical mask to the data"""
        return self._with_data_source(self._src._apply_mask(mask))

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

    def _generate_labels(self, lhs, operator, rhs, keep_unit):
        def get_label(item, key):
            return item.labels.get(key, "") if not np.isscalar(item) else str(item)

        labels = {"title": f"({get_label(lhs, 'title')} {operator} {get_label(rhs, 'title')})"}
        if keep_unit:
            labels["y"] = self.labels.get("y", "y")
        return labels

    def __neg__(self):
        if isinstance(self._src, TimeTags):
            raise NotImplementedError("This operation is not supported for TimeTag data")
        return Slice(
            self._src._with_data(-self.data),
            calibration=self._calibration,
            labels={
                "title": f"-{self.labels.get('title', 'title')}",
                "y": self.labels.get("y", "y"),
            },
        )

    def __add__(self, other):
        return Slice(
            self._src._with_data(self.data + self._unpack_other(other)),
            calibration=self._calibration if np.isscalar(other) else None,
            labels=self._generate_labels(self, "+", other, keep_unit=True),
        )

    def __sub__(self, other):
        return Slice(
            self._src._with_data(self.data - self._unpack_other(other)),
            calibration=self._calibration if np.isscalar(other) else None,
            labels=self._generate_labels(self, "-", other, keep_unit=True),
        )

    def __truediv__(self, other):
        return Slice(
            self._src._with_data(self.data / self._unpack_other(other)),
            labels=self._generate_labels(self, "/", other, keep_unit=False),
        )

    def __mul__(self, other):
        return Slice(
            self._src._with_data(self.data * self._unpack_other(other)),
            labels=self._generate_labels(self, "*", other, keep_unit=False),
        )

    def __pow__(self, other):
        return Slice(
            self._src._with_data(self.data ** self._unpack_other(other)),
            labels=self._generate_labels(self, "**", other, keep_unit=False),
        )

    def __rtruediv__(self, other):
        return Slice(
            self._src._with_data(self._unpack_other(other) / self.data),
            labels=self._generate_labels(other, "/", self, keep_unit=False),
        )

    def __rsub__(self, other):
        return Slice(
            self._src._with_data(self._unpack_other(other) - self.data),
            labels=self._generate_labels(other, "-", self, keep_unit=True),
        )

    def __rpow__(self, other):
        return Slice(
            self._src._with_data(self._unpack_other(other) ** self.data),
            labels=self._generate_labels(other, "**", self, keep_unit=False),
        )

    def __radd__(self, other):
        return Slice(
            self._src._with_data(self.data + self._unpack_other(other)),
            calibration=self._calibration if np.isscalar(other) else None,
            labels=self._generate_labels(other, "+", self, keep_unit=True),
        )

    def __rmul__(self, other):
        return Slice(
            self._src._with_data(self.data * self._unpack_other(other)),
            labels=self._generate_labels(other, "*", self, keep_unit=False),
        )

    @property
    def start(self):
        """Starting timestamp of this time series in nanoseconds"""
        return self._src.start

    @property
    def stop(self):
        """End timestamp of this time series in nanoseconds"""
        return self._src.stop

    @property
    def data(self) -> npt.ArrayLike:
        """The primary values of this channel slice"""
        return self._src.data

    @property
    def timestamps(self) -> npt.ArrayLike:
        """Absolute timestamps (since epoch) which correspond to the channel data"""
        return self._src.timestamps

    @property
    def seconds(self):
        """Relative time (in seconds) that corresponds to the channel data"""
        return (self._src.timestamps - self._src.start) * 1e-9

    @property
    def calibration(self) -> list:
        """List of force calibration items

        The first element represents the calibration item that was active when this slice was
        made.
        """
        if self._calibration:
            try:
                # Calibration data slicing is deferred until calibration is requested to avoid
                # slicing values that may be needed.
                return self._calibration.filter_calibration(self._src.start, self._src.stop)
            except IndexError:
                return []
        else:
            return []

    @property
    def sample_rate(self) -> Union[float, None]:
        """The data frequency for `Continuous` and `TimeSeries` data sources or `None` if it is not
        available or variable"""
        try:
            return self._src.sample_rate
        except AttributeError:
            return None

    @property
    def _timesteps(self) -> npt.ArrayLike:
        """The unique timesteps for `Continuous` and `TimeSeries` data sources"""
        try:
            return self._src._timesteps
        except AttributeError:
            raise NotImplementedError(
                f"`_timesteps` are not available for {self._src.__class__.__name__} data"
            )

    def downsampled_over(self, range_list, reduce=np.mean, where="center"):
        """Downsample channel data based on timestamp ranges.

        The downsampling function (e.g. np.mean) is evaluated for the time between a start and end
        time of each block. A :class:`Slice` is returned where each data point corresponds to the
        channel data downsampled over a block.

        Parameters
        ----------
        range_list : list of tuples
            A list of (start, stop) tuples indicating over which ranges to apply the function.
            Start and stop have to be specified in nanoseconds.
        reduce : callable
            The :mod:`numpy` function which is going to reduce multiple samples into one. The
            default is :func:`np.mean <numpy.mean>`, but :func:`np.sum <numpy.sum>` could also be
            appropriate for some cases, e.g. photon counts.
        where : str
            Where to put the final time point.

            - "center" : The new time points are set to
              `(timestamps_subset[0] + timestamps_subset[-1]) / 2`, where `timestamps_subset` are
              the timestamps corresponding to the samples being downsampled over.
            - "left" : Time points are set to the starting timestamp of the downsampled data.

        Returns
        -------
        slice : Slice
            A slice containing data that was downsampled over the desired time ranges.

        Raises
        ------
        TypeError
            If something other than a list is passed to `range_list`.
        ValueError
            If `range_list` does not contain a non-empty list of tuples of two elements (start and
            stop).
            If `where` is not set to either `"center"` or `"left"`.
        RuntimeError
            If there is no overlap between the time ranges specified in `range_list` and the
            :class:`Slice` which is to be downsampled.

        Examples
        --------
        ::

            from lumicks import pylake

            file = pylake.File("example.h5")
            stack = pylake.ImageStack("example.tiff")
            file.force1x.downsampled_over(stack.frame_timestamp_ranges())
        """
        if not isinstance(range_list, list):
            raise TypeError("Did not pass timestamps to range_list.")

        if len(range_list) == 0 or len(range_list[0]) != 2:
            raise ValueError("Did not pass timestamps to range_list.")

        if self._src.start >= range_list[-1][1] or self._src.stop <= range_list[0][0]:
            raise RuntimeError("No overlap between range and selected channel.")

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
            The :mod:`numpy` function which is going to reduce multiple samples into one. The
            default is :func:`np.mean <numpy.mean>`, but :func:`np.sum <numpy.sum>` could also be
            appropriate for some cases, e.g. photon counts.
        where : str
            Where to put the final time point.

            - "center" : The new time points are set to
              `(timestamps_subset[0] + timestamps_subset[-1]) / 2`, where `timestamps_subset` are
              the timestamps corresponding to the samples being downsampled over.
            - "left" : Time points are set to the starting timestamp of the downsampled data.

        method : str
            How to handle target sample times that are not exact multiples of the current sample
            time.

            - "safe" : New sample time must be an exact multiple of the current sample time, else
              an exception is raised.
            - 'ceil' : Rounds the sample rate up to the nearest frequency which fulfills this
              condition.
            - 'force' : Downsample data with the target input frequency; this will result in
              variable sample times and a variable number of sampling contributing to each target
              sample but must be used for variable-frequency data.
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
            remainder = target_timestep % source_timestep[0]
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
            The :mod:`numpy` function which is going to reduce multiple samples into one. The
            default is :func:`np.mean <numpy.mean>`, but :func:`np.sum <numpy.sum>` could also be
            appropriate for some cases, e.g. photon counts.
        """
        return self._with_data_source(self._src.downsampled_by(factor, reduce))

    def downsampled_like(self, other_slice, reduce=np.mean):
        """Downsample high frequency data analogously to a low frequency channel in the same way
        that Bluelake does it.

        Note: some data required to reconstruct the first low frequency time point can actually
        occur before the starting timestamp of the marker and is therefore missing from the
        exported `.h5` file. Therefore, it is not always possible to downsample to all of the data
        points in the low frequency `other_slice`. This function returns both the requested
        downsampled channel data *and* a copy of the input channel cropped such that both
        returned :class:`~lumicks.pylake.channel.Slice` objects have the same time points.

        Parameters
        ----------
        other_slice : Slice
            Timeline channel to downsample like. This should be a low frequency channel that
            provides the timestamps to downsample to.
        reduce : callable
            The :mod:`numpy` function which is going to reduce multiple samples into one. The
            default is :func:`np.mean <numpy.mean>`, but :func:`np.sum <numpy.sum>` could also be
            appropriate for some cases, e.g. photon counts.

        Returns
        -------
        downsampled_slice : Slice
            This channel downsampled to the same timestamp ranges as `other_slice`.
        cropped_other_slice : Slice
            A copy of `other_slice` cropped such that the timestamps match those of
            `downsampled_slice`.

        Raises
        ------
        TypeError
            If the other slice is not a low frequency channel slice.
        NotImplementedError
            If this slice is a low frequency channel slice.
        RuntimeError
            If there is no overlap between the two slices.
        """
        if not isinstance(other_slice._src, TimeSeries):
            raise TypeError(
                "You did not pass a low frequency channel to serve as reference channel."
            )

        if not isinstance(self._src, Continuous):
            raise NotImplementedError(
                "Downsampled_like is only available for high frequency channels"
            )

        timestamps = other_slice.timestamps
        delta_time = np.diff(timestamps)

        if self._src.start > (timestamps[-1] - delta_time[-1]) or self._src.stop <= timestamps[0]:
            raise RuntimeError("No overlap between slices.")

        # When the frame rate changes, one frame is very long due to the delay of the camera. It
        # should default to the new frame rate.
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

    def range_selector(self, show=True, **kwargs) -> SliceRangeSelectorWidget:
        """Show a range selector widget

        Opens a widget used to select time ranges. The timestamps of these time ranges can then be
        extracted from
        :attr:`selector.ranges <lumicks.pylake.nb_widgets.range_selector.SliceRangeSelectorWidget.ranges>`,
        while the slices can be extracted from
        :attr:`selector.slices <lumicks.pylake.nb_widgets.range_selector.SliceRangeSelectorWidget.slices>`.

        Actions
        -------
        left-click
            Define time ranges by clicking the left and then the right boundary of the region you
            wish to select.
        right-click
            Remove previously selected time range.

        Parameters
        ----------
        show : bool, optional
            Show the plot. Default is True.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        SliceRangeSelectorWidget
        """
        return SliceRangeSelectorWidget(self, show=show, **kwargs)


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
        self.dt = dt  # ns

    def _with_data(self, data):
        return self.__class__(data, self.start, self.dt)

    def _apply_mask(self, mask):
        if len(mask) != len(self.data):
            raise IndexError("Length of the logical mask did not match length of the data.")
        return TimeSeries(self.data[mask], self.timestamps[mask])

    def __len__(self):
        return len(self._src_data)

    @staticmethod
    def from_dataset(dset, y_label="y", calibration=None):
        start = dset.attrs["Start time (ns)"]
        dt = int(1e9 / dset.attrs["Sample rate (Hz)"])  # ns
        return Slice(
            Continuous(dset, start, dt),
            labels={"title": dset.name.strip("/"), "y": y_label},
            calibration=calibration,
        )

    def to_dataset(self, parent, name, **kwargs):
        """Save this to an h5 dataset

        Parameters
        ----------
        parent : h5py.Group or h5py.File
            location to save to.
        name : str
            name of the new dataset
        **kwargs
            forwarded to h5py.Group.create_dataset()
        """
        dset = parent.create_dataset(name, data=self._src_data, **kwargs)
        dset.attrs["Kind"] = "Continuous"
        dset.attrs["Sample rate (Hz)"] = self.sample_rate
        dset.attrs["Start time (ns)"] = self.start
        dset.attrs["Stop time (ns)"] = self.stop
        return dset

    @property
    def data(self) -> npt.ArrayLike:
        if self._cached_data is None:
            self._cached_data = np.asarray(self._src_data)
        return self._cached_data

    @property
    def timestamps(self) -> npt.ArrayLike:
        return np.arange(self.start, self.stop, self.dt)

    @property
    def sample_rate(self) -> float:
        return 1e9 / self.dt  # Hz

    @property
    def _timesteps(self) -> npt.ArrayLike:
        return np.asarray([self.dt], dtype=np.int64)  # ns

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

    Raises
    ------
    ValueError
        If the number of data points and the number of timestamps are unequal.
    """

    def __init__(self, data, timestamps):
        if len(data) != len(timestamps):
            raise ValueError(
                f"Number of data points ({len(data)}) should be the same as number of timestamps "
                f"({len(timestamps)})."
            )

        self._src_data = data
        self._cached_data = None
        self._src_timestamps = timestamps
        self._cached_timestamps = None

    def __len__(self):
        return len(self._src_data)

    def _with_data(self, data):
        return self.__class__(data, self.timestamps)

    def _apply_mask(self, mask):
        if len(mask) != len(self.data):
            raise IndexError("Length of the logical mask did not match length of the data.")
        return TimeSeries(self.data[mask], self.timestamps[mask])

    @staticmethod
    def from_dataset(dset, y_label="y", calibration=None) -> Slice:
        class LazyLoadedCompoundField:
            """Wrapper to enable lazy loading of HDF5 compound datasets

            Notes
            -----
            We only need to support the methods `__array__()` and `__len__()`, as we only access
            `LazyLoadedCompoundField` via the properties `TimeSeries.data`, `timestamps` and the
            method `__len__()`.

            `LazyLoadCompoundField` might be replaced with `dset.fields(fieldname)` if and when the
            returned `FieldsWrapper` object provides an `__array__()` method itself"""

            def __init__(self, dset, fieldname):
                self._dset = dset
                self._fieldname = fieldname

            def __array__(self):
                """Get the data of the field as an array"""
                return self._dset[self._fieldname]

            def __len__(self):
                """Get the length of the underlying dataset"""
                return len(self._dset)

        data = LazyLoadedCompoundField(dset, "Value")
        timestamps = LazyLoadedCompoundField(dset, "Timestamp")
        return Slice(
            TimeSeries(data, timestamps),
            labels={"title": dset.name.strip("/"), "y": y_label},
            calibration=calibration,
        )

    def to_dataset(self, parent, name, **kwargs):
        """Save this to an h5 dataset

        Parameters
        ----------
        parent : h5py.Group or h5py.File
            location to save to.
        name : str
            name of the new dataset
        **kwargs
            forwarded to h5py.Group.create_dataset()
        """
        compound_type = np.dtype([("Timestamp", np.int64), ("Value", float)])
        data = np.array([(t, d) for t, d in zip(self.timestamps, self.data)], compound_type)
        dset = parent.create_dataset(name, data=data, **kwargs)
        dset.attrs["Kind"] = b"TimeSeries"
        dset.attrs["Start time (ns)"] = self.start
        dset.attrs["Stop time (ns)"] = self.stop
        return dset

    @property
    def data(self) -> npt.ArrayLike:
        if self._cached_data is None:
            self._cached_data = np.asarray(self._src_data)
        return self._cached_data

    @property
    def timestamps(self) -> npt.ArrayLike:
        if self._cached_timestamps is None:
            self._cached_timestamps = np.asarray(self._src_timestamps)
        return self._cached_timestamps

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
    def sample_rate(self) -> Union[float, None]:
        """The data frequency or `None` if it is variable"""
        timestep = self._timesteps
        if len(timestep) == 1:
            return 1e9 / timestep[0]  # Hz

    @property
    def _timesteps(self) -> npt.ArrayLike:
        return np.unique(np.diff(self.timestamps))  # ns

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

    def _apply_mask(self, mask):
        raise NotImplementedError("Time tags do not currently support this operation")

    @staticmethod
    def from_dataset(dset, y_label="y"):
        return Slice(TimeTags(dset))

    def to_dataset(self, parent, name, **kwargs):
        """Save this to an h5 dataset

        Parameters
        ----------
        parent : h5py.Group or h5py.File
            location to save to.
        name : str
            name of the new dataset
        **kwargs
            forwarded to h5py.Group.create_dataset()
        """
        dset = parent.create_dataset(name, data=self.data, **kwargs)
        dset.attrs["Kind"] = "TimeTags"
        return dset

    @property
    def timestamps(self) -> npt.ArrayLike:
        # For time tag data, the data is the timestamps!
        return self.data

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
    def data(self) -> npt.ArrayLike:
        return np.empty(0)

    @property
    def timestamps(self) -> npt.ArrayLike:
        return np.empty(0)

    @property
    def start(self):
        return None

    @property
    def stop(self):
        return None


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
