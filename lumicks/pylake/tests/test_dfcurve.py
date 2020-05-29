import numpy as np

from lumicks.pylake.fdcurve import FdCurve
from lumicks.pylake.channel import Slice, TimeSeries


def make_mock_fd(force, distance, start=0):
    """Mock Fd curve which is not attached to an actual file, timestamps start at `start`"""
    assert len(force) == len(distance)
    fd = FdCurve(file=None, start=None, stop=None, name="")
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
