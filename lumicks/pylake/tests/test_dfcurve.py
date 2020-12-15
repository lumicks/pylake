import numpy as np
import pytest

from lumicks.pylake.fdcurve import FDCurve
from lumicks.pylake.channel import Slice, TimeSeries


def make_mock_fd(force, distance, start=0):
    """Mock FD curve which is not attached to an actual file, timestamps start at `start`"""
    assert len(force) == len(distance)
    fd = FDCurve(file=None, start=None, stop=None, name="")
    timestamps = np.arange(len(force)) + start
    fd._force_cache = Slice(TimeSeries(force, timestamps))
    fd._distance_cache = Slice(TimeSeries(distance, timestamps))
    return fd


def test_subtraction():
    fd1 = make_mock_fd(force=[1, 2, 3], distance=[0, 1, 2], start=0)
    fd2 = make_mock_fd(force=[2, 2, 2], distance=[0, 1, 2], start=100)
    assert np.allclose((fd1 - fd2).f.data, [-1, 0, 1])
    assert np.allclose((fd2 - fd1).f.data, [1, 0, -1])

    fd1 = make_mock_fd(force=[1, 2, 3], distance=[0, 1, 2], start=0)
    fd2 = make_mock_fd(force=[2, 2, 2], distance=[1, 2, 3], start=100)
    assert np.allclose((fd1 - fd2).f.data, [0, 1])
    assert np.allclose((fd2 - fd1).f.data, [0, -1])

    fd1 = make_mock_fd(force=[1, 2, 3], distance=[0, 1, 2], start=0)
    fd2 = make_mock_fd(force=[1, 1, 1], distance=[5, 6, 7], start=100)
    assert np.allclose((fd1 - fd2).f.data, [])
    assert np.allclose((fd2 - fd1).f.data, [])


def test_slice():
    fd = make_mock_fd(force=[1, 1, 1], distance=[5, 6, 7], start=100)
    assert np.allclose(fd[101:].d.data, [6, 7])
    assert np.allclose(fd[:101].d.data, [5])
    assert np.allclose(fd[101:300].d.data, [6, 7])
    assert np.allclose(fd[101:102].d.data, [6])

    sliced = fd[101:102]
    assert id(fd.f) != id(sliced.f)
    assert id(fd.d) != id(sliced.d)


@pytest.mark.parametrize("field,changed_var,other_var", [
    ("force_offset", "f", "d"),
    ("distance_offset", "d", "f")
])
def test_offsets(field, changed_var, other_var):
    fd1 = make_mock_fd(force=[1, 2, 3], distance=[2, 3, 4], start=0)

    # Make sure the cache is created
    fd1.f
    fd1.d

    fd1_sub = fd1.with_offset(**{field: -1})
    assert fd1_sub is not fd1
    assert fd1_sub.f is not fd1.f
    assert fd1_sub.d is not fd1.d
    assert fd1_sub.f.data is not fd1.f.data
    assert fd1_sub.d.data is not fd1.d.data
    assert np.allclose(getattr(fd1_sub, changed_var).data, getattr(fd1, changed_var).data - 1)
    assert np.allclose(getattr(fd1_sub, other_var).data, getattr(fd1, other_var).data)


def test_distance_offset_tracking_lost():
    """A value of 0 means the tracking was lost. This value should not change when subtracting baselines. An
    unfortunate side effect of using a regular number for this is that when subtraction leads to a distance of zero,
    this will lead to that data becoming a missing value point. This should not happen for real data though."""
    fd1 = make_mock_fd(force=[1, 2, 3], distance=[2, 0, 4], start=0)
    sub = fd1.with_offset(distance_offset=-1)
    assert np.allclose(sub.d.data, [1, 0, 3])


def test_subtract_too_much_distance():
    """Tests whether we get an exception when we subtract more than the lowest valid distance value"""
    fd1 = make_mock_fd(force=[1, 2, 3], distance=[2.0, 0.0, 4.0], start=0)

    np.allclose(fd1.with_offset(distance_offset=-1.0).d.data, [1.0, 0.0, 3.0])

    with pytest.raises(ValueError):
        fd1.with_offset(distance_offset=-2.0)

    with pytest.raises(ValueError):
        fd1.with_offset(distance_offset=-4.0)


@pytest.mark.parametrize("slice_parameters,f,d", [
    ({}, np.arange(-0.5, 3, 0.5), np.arange(0.5, 4, .5)),
    ({"force_max": 1}, np.arange(-0.5, 1, 0.5), np.arange(0.5, 2.0, .5)),
    ({"force_min": 0}, np.arange(0, 3, 0.5), np.arange(1.0, 4, .5)),
    ({"distance_min": 1}, np.arange(0.0, 3, 0.5), np.arange(1.0, 4, .5)),
    ({"distance_max": 3}, np.arange(-0.5, 2, 0.5), np.arange(0.5, 3, .5)),
])
def test_fd_slice(slice_parameters, f, d):
    fd1 = make_mock_fd(force=np.arange(-2, 3, .5), distance=np.arange(-1, 4, .5), start=0)

    fdr = fd1.sliced(**slice_parameters)
    assert np.allclose(fdr.f, f)
    assert np.allclose(fdr.d, d)
