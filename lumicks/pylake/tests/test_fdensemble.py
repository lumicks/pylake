import numpy as np
from lumicks.pylake.detail.alignment import align_force_simple, align_distance_simple, align_fd_simple
from lumicks.pylake.fdcurve import FdCurve
from lumicks.pylake.channel import Slice, TimeSeries
from lumicks.pylake.fdensemble import FdEnsemble


def make_mock_fd(force, distance, start=0):
    """Mock FD curve which is not attached to an actual file, timestamps start at `start`"""
    assert len(force) == len(distance)
    fd = FdCurve(file=None, start=None, stop=None, name="")
    timestamps = np.arange(len(force)) + start
    fd._force_cache = Slice(TimeSeries(force, timestamps))
    fd._distance_cache = Slice(TimeSeries(distance, timestamps))
    return fd


def test_align_force_simple():
    """Test whether alignment of curves with a constant offset in force produces aligned curves"""

    force = np.hstack((np.ones(50), np.arange(1, 50)))
    distance = np.arange(1, 100)
    fd1 = make_mock_fd(force=force, distance=distance, start=0)
    fd2 = make_mock_fd(force=force + 5, distance=distance, start=0)
    fd3 = make_mock_fd(force=force + 15, distance=distance, start=0)
    aligned = align_force_simple({"fd1": fd1, "fd2": fd2, "fd3": fd3}, distance_range=50)

    for fd in aligned.values():
        np.testing.assert_allclose(fd.d.data, distance)
        np.testing.assert_allclose(fd.f.data, force)


def test_align_distance_simple():
    """Test whether alignment of curves with a constant offset in distance produces aligned curves"""

    force = np.hstack((np.ones(50), np.arange(1, 52)))
    distance = np.arange(1.0, 102.0)
    fd1 = make_mock_fd(force=force, distance=distance, start=0)
    fd2 = make_mock_fd(force=force, distance=distance + 4.0, start=0)
    fd3 = make_mock_fd(force=force, distance=distance + 2.0, start=0)
    aligned = align_distance_simple({"fd1": fd1, "fd2": fd2, "fd3": fd3}, distance_range=50)

    for fd in aligned.values():
        np.testing.assert_allclose(fd.d.data, distance)
        np.testing.assert_allclose(fd.f.data, force)


def test_align_fd_simple():
    """Test whether alignment of curves with a constant offset in both force and distance produces aligned curves"""

    force = np.hstack((np.ones(50), np.arange(1, 52)))
    distance = np.arange(1.0, 102.0)
    fd1 = make_mock_fd(force=force, distance=distance, start=0)
    fd2 = make_mock_fd(force=force + 5.0, distance=distance + 4.0, start=0)
    fd3 = make_mock_fd(force=force + 15.0, distance=distance + 2.0, start=0)
    aligned = align_fd_simple({"fd1": fd1, "fd2": fd2, "fd3": fd3}, 50, 50)

    for fd in aligned.values():
        np.testing.assert_allclose(fd.f.data, force)
        np.testing.assert_allclose(fd.d.data, distance)


def test_non_constant_rate_fd_alignment_simple():
    """Tests what happens when we try to align an Fd curve with a non-constant pulling rate. A non-constant pulling rate
    would lead to a non-linear relation when just looking at either force or distance, rather than looking at them
    at the same time. Here we test whether the alignment actually operates on f,d rather than on channels
    independently"""

    distance = np.hstack((np.arange(150) * np.arange(150)) / 2500)
    force = np.copy(distance)
    force[force < 2] = 2

    def generate_data(num_shortened):
        return make_mock_fd(force=force[:-num_shortened], distance=distance[:-num_shortened], start=0)

    shortening = [10, 20, 33, 48, 14, 60]
    fds = {f"f_{idx}": generate_data(num_shortened) for idx, num_shortened in enumerate(shortening)}
    aligned = align_fd_simple(fds, 1.5, 1.2)

    for fd, num_shortened in zip(aligned.values(), shortening):
        np.testing.assert_allclose(fd.f.data, force[:-num_shortened])
        np.testing.assert_allclose(fd.d.data, distance[:-num_shortened])


def test_back_and_forth_rate_fd_alignment_simple():
    """Test what happens when we align an F,d curve that keeps going back and forth (non monotonic distance). The reason
    we want to test this specific case is because there may be F,d curves that go back and forth. If the code merely
    takes the last and first N samples, then this test would fail."""
    
    distance = 2.0 + np.sin(np.arange(1000))
    force = np.copy(distance)
    force[force < 2.0] = 2

    def generate_data(num_shortened):
        return make_mock_fd(force=force[:-num_shortened], distance=distance[:-num_shortened], start=0)

    shortening = [100, 200, 330]
    fds = {f"fd_{idx}": generate_data(num_shortened) for idx, num_shortened in enumerate(shortening)}
    aligned = align_fd_simple(fds, 1, 1)

    for fd, num_shortened in zip(aligned.values(), shortening):
        np.testing.assert_allclose(fd.f.data, force[:-num_shortened])
        np.testing.assert_allclose(fd.d.data, distance[:-num_shortened])


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

    # Test that alignment leads to new objects and that the originals weren't modified
    fd_ensemble.align_linear(1, 1)
    for fd_a, fd_new, fd_unchanged in zip(fds.values(), fd_ensemble.values(), fd_ensemble.raw.values()):
        assert id(fd_a) != id(fd_new)
        assert id(fd_a) == id(fd_unchanged)


def test_fd_ensemble_accessors():
    force = np.arange(3)
    distance = np.arange(3)
    fds = {
        "fd1": make_mock_fd(force=force, distance=distance, start=0),
        "fd2": make_mock_fd(force=force + 5, distance=distance, start=0),
        "fd3": make_mock_fd(force=force + 15, distance=distance, start=0),
    }
    fd_ensemble = FdEnsemble(fds)

    np.testing.assert_allclose(fd_ensemble.f, np.array([0, 1, 2, 5, 6, 7, 15, 16, 17]))
    np.testing.assert_allclose(fd_ensemble.d, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]))

    fd_ensemble.align_linear(20, 20)
    np.testing.assert_allclose(fd_ensemble.f, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]))
