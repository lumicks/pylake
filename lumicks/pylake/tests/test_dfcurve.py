import numpy as np

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
