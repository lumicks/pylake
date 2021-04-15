import pytest
import numpy as np
from lumicks.pylake.channel import Slice, Continuous, TimeSeries, TimeTags

t_start = 1 + int(1e18)
time_series = np.array([1, 2, 3, 4, 5], dtype=np.int64) + int(1e18)
slice_continuous_1 = Slice(Continuous([1, 2, 3, 4, 5], start=t_start, dt=1))
slice_continuous_2 = Slice(Continuous([2, 2, 2, 2, 2], start=t_start, dt=1))
slice_timeseries_1 = Slice(TimeSeries([1, 2, 3, 4, 5], time_series))
slice_timeseries_2 = Slice(TimeSeries([2, 2, 2, 2, 2], time_series))

operators = [
    "__add__",
    "__sub__",
    "__truediv__",
    "__mul__",
    "__pow__",
    "__radd__",
    "__rsub__",
    "__rtruediv__",
    "__rmul__",
    "__rpow__",
]


@pytest.mark.parametrize(
    "slice1, slice2",
    [
        (slice_continuous_1, slice_continuous_2),
        (slice_timeseries_1, slice_timeseries_2),
        (slice_continuous_1, slice_timeseries_2),
        (slice_timeseries_1, slice_continuous_2),
    ],
)
def test_operations_slice(slice1, slice2):
    def test_operator(first_slice, second_slice, operation):
        # Operations in the containers must give the same result as operating on the data directly.
        assert np.array_equal(
            getattr(first_slice, operation)(second_slice).data,
            getattr(first_slice.data, operation)(second_slice.data),
        )

        # Operations shouldn't modify the timestamps.
        assert np.array_equal(
            getattr(first_slice, operation)(second_slice).timestamps, first_slice.timestamps
        )

    for operator in operators:
        test_operator(slice1, slice2, operator)


@pytest.mark.parametrize(
    "slice1, scalar",
    [
        (slice_continuous_1, 2.0),
        (slice_timeseries_1, 2.0),
    ],
)
def test_operations_scalar(slice1, scalar):
    def test_operator(current_slice, scalar_val, operation):
        # Operations in the containers must give the same result as operating on the data directly.
        assert np.array_equal(
            getattr(current_slice, operation)(scalar_val).data,
            getattr(current_slice.data, operation)(scalar_val),
        )

        # Operations shouldn't modify the timestamps.
        assert np.array_equal(
            getattr(current_slice, operation)(scalar_val).timestamps, current_slice.timestamps
        )

    for operator in operators:
        test_operator(slice1, scalar, operator)


slice_continuous_different_timestamps = Slice(Continuous([2, 2, 2, 2, 2], start=t_start + 1, dt=1))
slice_timeseries_different_timestamps = Slice(TimeSeries([2, 2, 2, 2, 2], time_series + 1))


@pytest.mark.parametrize(
    "slice1, slice2",
    [
        (slice_continuous_1, slice_continuous_different_timestamps),
        (slice_timeseries_1, slice_timeseries_different_timestamps),
        (slice_continuous_1, slice_timeseries_different_timestamps),
        (slice_timeseries_1, slice_continuous_different_timestamps),
    ],
)
def test_incompatible_timestamps(slice1, slice2):
    # Test whether the timestamps are correctly compared
    for operator in operators:
        with pytest.raises(RuntimeError):
            getattr(slice1, operator)(slice2)


slice_continuous_different_length = Slice(Continuous([2, 2, 2, 2], start=t_start, dt=1))
slice_timeseries_different_length = Slice(TimeSeries([2, 2, 2, 2], [1, 2, 3, 4]))


@pytest.mark.parametrize(
    "slice1, slice2",
    [
        (slice_continuous_1, slice_continuous_different_length),
        (slice_timeseries_1, slice_timeseries_different_length),
        (slice_continuous_1, slice_timeseries_different_length),
        (slice_timeseries_1, slice_continuous_different_length),
    ],
)
def test_incompatible_length(slice1, slice2):
    # The data sets need to be the same length. We explicitly raise an exception when they are not.
    for operator in operators:
        with pytest.raises(RuntimeError):
            getattr(slice1, operator)(slice2)


@pytest.mark.parametrize(
    "slice1, slice2",
    [
        (slice_continuous_1, [2, 2, 2, 2, 2]),
        (slice_timeseries_1, [2, 2, 2, 2, 2]),
        (slice_continuous_1, np.array([2, 2, 2, 2, 2])),
        (slice_timeseries_1, np.array([[2, 2, 2, 2, 2]])),
    ],
)
def test_incompatible_types(slice1, slice2):
    for operator in operators:
        with pytest.raises(TypeError):
            getattr(slice1, operator)(slice2)


timetags = Slice(TimeTags(time_series))


@pytest.mark.parametrize(
    "data1, data2",
    [
        (slice_continuous_1, timetags),
        (slice_timeseries_1, timetags),
        (timetags, slice_continuous_2),
        (timetags, slice_timeseries_2),
    ],
)
def test_timetags_not_implemented(data1, data2):
    # This is not implemented for time tags at this point (we need to determine what are sensible
    # operations on those first).
    for operator in operators:
        with pytest.raises(NotImplementedError):
            getattr(data1, operator)(data2)
