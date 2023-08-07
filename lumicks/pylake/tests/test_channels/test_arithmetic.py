import numpy as np
import pytest

from lumicks.pylake.channel import Slice, TimeTags, Continuous, TimeSeries
from lumicks.pylake.calibration import ForceCalibration

start = 1 + int(1e18)
calibration = ForceCalibration("Stop time (ns)", [{"Stop time (ns)": start, "kappa (pN/nm)": 0.45}])
time_series = np.array([1, 2, 3, 4, 5], dtype=np.int64) + int(1e18)
slice_continuous_1 = Slice(Continuous([1, 2, 3, 4, 5], start=start, dt=1), calibration=calibration)
slice_continuous_2 = Slice(Continuous([2, 2, 2, 2, 2], start=start, dt=1), calibration=calibration)
slice_timeseries_1 = Slice(TimeSeries([1, 2, 3, 4, 5], time_series), calibration=calibration)
slice_timeseries_2 = Slice(TimeSeries([2, 2, 2, 2, 2], time_series), calibration=calibration)

# Operators, whether they preserve force calibration when operated with a scalar,
# their string representation and whether they keep their unit under this operation
operators = [
    ["__add__", True, "+", True],
    ["__sub__", True, "-", True],
    ["__truediv__", False, "/", False],
    ["__mul__", False, "*", False],
    ["__pow__", False, "**", False],
    ["__radd__", True, "+", True],
    ["__rsub__", False, "-", True],
    ["__rtruediv__", False, "/", False],
    ["__rmul__", False, "*", False],
    ["__rpow__", False, "**", False],
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

        # With slices, force calibration should never be carried over (since it's invalid)
        assert not getattr(first_slice, operation)(second_slice).calibration

    for operator, *_ in operators:
        test_operator(slice1, slice2, operator)


@pytest.mark.parametrize(
    "slice1, scalar",
    [
        (slice_continuous_1, 2.0),
        (slice_timeseries_1, 2.0),
    ],
)
def test_operations_scalar(slice1, scalar):
    def test_operator(current_slice, scalar_val, operation, preserve_calibration):
        # Operations in the containers must give the same result as operating on the data directly.
        assert np.array_equal(
            getattr(current_slice, operation)(scalar_val).data,
            getattr(current_slice.data, operation)(scalar_val),
        )

        # Operations shouldn't modify the timestamps.
        assert np.array_equal(
            getattr(current_slice, operation)(scalar_val).timestamps, current_slice.timestamps
        )

        if preserve_calibration:
            assert len(getattr(current_slice, operation)(scalar_val).calibration) == 1
        else:
            assert not getattr(current_slice, operation)(scalar_val).calibration

    for operator, preserve_calibration, *_ in operators:
        test_operator(slice1, scalar, operator, preserve_calibration)


slice_continuous_different_timestamps = Slice(
    Continuous([2, 2, 2, 2, 2], start=start + 1, dt=1), calibration=calibration
)
slice_timeseries_different_timestamps = Slice(
    TimeSeries([2, 2, 2, 2, 2], time_series + 1), calibration=calibration
)


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
    for operator, *_ in operators:
        with pytest.raises(RuntimeError):
            getattr(slice1, operator)(slice2)


slice_continuous_different_length = Slice(
    Continuous([2, 2, 2, 2], start=start, dt=1), calibration=calibration
)
slice_timeseries_different_length = Slice(
    TimeSeries([2, 2, 2, 2], [1, 2, 3, 4]), calibration=calibration
)


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
    for operator, *_ in operators:
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
    for operator, *_ in operators:
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
    for operator, *_ in operators:
        with pytest.raises(NotImplementedError):
            getattr(data1, operator)(data2)


@pytest.mark.parametrize("channel_slice", [(slice_continuous_1), (slice_timeseries_1)])
def test_negation(channel_slice):
    neg_slice = -channel_slice
    np.testing.assert_allclose(neg_slice.data, -channel_slice.data)
    np.testing.assert_allclose(neg_slice.timestamps, channel_slice.timestamps)
    assert len(neg_slice.calibration) == 1


def test_negation_timetags_not_implemented():
    with pytest.raises(NotImplementedError):
        negated_timetags = -timetags


def test_labels_slices():
    """Test whether the plot labels are appropriately constructed if valid"""
    x = Slice(Continuous([1, 2, 3], start=start, dt=1), labels={"y": "Force [pN]", "title": "x"})
    y = Slice(Continuous([1, 2, 3], start=start, dt=1), labels={"y": "Force [pN]", "title": "y"})

    for operation, _, operator_str, keep_dims in operators:
        if operation.startswith("__r"):
            assert getattr(x, operation)(y).labels["title"] == f"(y {operator_str} x)"
        else:
            assert getattr(x, operation)(y).labels["title"] == f"(x {operator_str} y)"

        if keep_dims:
            assert getattr(x, operation)(y).labels["y"] == "Force [pN]"
        else:
            assert "y" not in getattr(x, operation)(y).labels


def test_labels_scalars():
    """Test whether the plot labels are appropriately constructed if valid"""
    x = Slice(Continuous([1, 2, 3], start=start, dt=1), labels={"y": "Force [pN]", "title": "x"})
    y = 5

    for operation, _, operator_str, keep_dims in operators:
        if operation.startswith("__r"):
            assert getattr(x, operation)(y).labels["title"] == f"(5 {operator_str} x)"
        else:
            assert getattr(x, operation)(y).labels["title"] == f"(x {operator_str} 5)"

        if keep_dims:
            assert getattr(x, operation)(y).labels["y"] == "Force [pN]"
        else:
            assert "y" not in getattr(x, operation)(y).labels


def test_negation_label():
    x = Slice(Continuous([1, 2, 3], start=start, dt=1), labels={"y": "Force [pN]", "title": "x"})
    assert (-x).labels["title"] == "-x"
    assert (-x).labels["y"] == "Force [pN]"
