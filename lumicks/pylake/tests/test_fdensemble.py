import pytest
import numpy as np
from lumicks.pylake.detail.alignment import align_force_simple, align_distance_simple, align_fd_simple
from lumicks.pylake.fdcurve import FDCurve
from lumicks.pylake.channel import Slice, TimeSeries
from lumicks.pylake.fdensemble import FdEnsemble


def make_mock_fd(force, distance, start=0):
    """Mock FD curve which is not attached to an actual file, timestamps start at `start`"""
    assert len(force) == len(distance)
    fd = FDCurve(file=None, start=None, stop=None, name="")
    timestamps = np.arange(len(force)) + start
    fd._force_cache = Slice(TimeSeries(force, timestamps))
    fd._distance_cache = Slice(TimeSeries(distance, timestamps))
    return fd


def test_iteration_fd_ensemble():
    """Test whether we can iterate over the ensemble"""
    force = np.hstack((np.ones(50), np.arange(1, 52)))
    distance = np.hstack((np.arange(1, 102)))
    fds = {
        "fd1": make_mock_fd(force=force, distance=distance, start=0),
        "fd2": make_mock_fd(force=force + 5, distance=distance, start=0),
        "fd3": make_mock_fd(force=force + 15, distance=distance, start=0),
    }
    fd_ensemble = FdEnsemble(fds)

    for key1, key2 in zip(fds, fd_ensemble):
        assert key1 == key2

    for key1, key2 in zip(fds.items(), fd_ensemble.items()):
        assert key1 == key2

    # The curves should be the same
    for fd_a, fd_b in zip(fds.values(), fd_ensemble.values()):
        assert id(fd_a) == id(fd_b)

    for fd_a, fd_b in zip(fds.values(), fd_ensemble.raw.values()):
        assert id(fd_a) == id(fd_b)


def test_fd_ensemble_accessors():
    force = np.arange(3)
    distance = np.arange(3)
    fds = {
        "fd1": make_mock_fd(force=force, distance=distance, start=0),
        "fd2": make_mock_fd(force=force + 5, distance=distance, start=0),
        "fd3": make_mock_fd(force=force + 15, distance=distance, start=0),
    }
    fd_ensemble = FdEnsemble(fds)

    assert np.allclose(fd_ensemble.f, np.array([0, 1, 2, 5, 6, 7, 15, 16, 17]))
    assert np.allclose(fd_ensemble.d, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]))
