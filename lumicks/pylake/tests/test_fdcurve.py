import numpy as np
import pytest

from lumicks.pylake import File
from lumicks.pylake.fdcurve import FdCurve
from lumicks.pylake.channel import Slice, TimeSeries


def make_mock_fd(force, distance, start=0, file=None):
    """Mock FD curve which is not attached to an actual file, timestamps start at `start`"""
    assert len(force) == len(distance)
    fd = FdCurve(file=file, start=None, stop=None, name="")
    timestamps = np.arange(len(force)) + start
    fd._force_cache = Slice(TimeSeries(force, timestamps))
    fd._distance_cache = Slice(TimeSeries(distance, timestamps))
    return fd


def test_subtraction():
    fd1 = make_mock_fd(force=[1, 2, 3], distance=[0, 1, 2], start=0)
    fd2 = make_mock_fd(force=[2, 2, 2], distance=[0, 1, 2], start=100)
    np.testing.assert_allclose((fd1 - fd2).f.data, [-1, 0, 1])
    np.testing.assert_allclose((fd2 - fd1).f.data, [1, 0, -1])

    fd1 = make_mock_fd(force=[1, 2, 3], distance=[0, 1, 2], start=0)
    fd2 = make_mock_fd(force=[2, 2, 2], distance=[1, 2, 3], start=100)
    np.testing.assert_allclose((fd1 - fd2).f.data, [0, 1])
    np.testing.assert_allclose((fd2 - fd1).f.data, [0, -1])

    fd1 = make_mock_fd(force=[1, 2, 3], distance=[0, 1, 2], start=0)
    fd2 = make_mock_fd(force=[1, 1, 1], distance=[5, 6, 7], start=100)
    np.testing.assert_allclose((fd1 - fd2).f.data, [])
    np.testing.assert_allclose((fd2 - fd1).f.data, [])


def test_slice():
    fd = make_mock_fd(force=[1, 1, 1], distance=[5, 6, 7], start=100)
    np.testing.assert_allclose(fd[101:].d.data, [6, 7])
    np.testing.assert_allclose(fd[:101].d.data, [5])
    np.testing.assert_allclose(fd[101:300].d.data, [6, 7])
    np.testing.assert_allclose(fd[101:102].d.data, [6])

    sliced = fd[101:102]
    assert id(fd.f) != id(sliced.f)
    assert id(fd.d) != id(sliced.d)


def test_sliced_by_distance():
    # test slicing
    d = np.arange(10)
    fd = make_mock_fd(force=np.ones(d.shape), distance=d)
    # beginning
    sub = fd._sliced_by_distance(0, 4, max_gap=0)
    assert np.all(np.equal([0, 1, 2, 3, 4], sub.d.timestamps))
    sub = fd._sliced_by_distance(-1, 4, max_gap=0)
    assert np.all(np.equal([0, 1, 2, 3, 4], sub.d.timestamps))
    # middle
    sub = fd._sliced_by_distance(3, 6, max_gap=0)
    assert np.all(np.equal([3, 4, 5, 6], sub.d.timestamps))
    # end
    sub = fd._sliced_by_distance(7, 9, max_gap=0)
    assert np.all(np.equal([7, 8, 9], sub.d.timestamps))
    sub = fd._sliced_by_distance(7, 11, max_gap=0)
    assert np.all(np.equal([7, 8, 9], sub.d.timestamps))

    # test noise handling
    min_dist = 1.5
    max_dist = 2
    max_gap = 2

    # "ideal" data
    d0 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                   3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    f0 = np.ones(d0.shape)
    fd0 = make_mock_fd(force=f0, distance=d0)

    # find timestamps of proper slice
    is_in_range = np.logical_and(min_dist <= d0, d0 <= max_dist)#.astype(np.int)
    ts_in_range = fd0.d.timestamps[is_in_range]

    # "blip" before changepoint - within gap limit
    d1 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.1, 2.0,
                   3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    # "blip" before changepoint - outside gap limit
    d2 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.1, 2.0, 2.0, 2.0, 2.0,
                   3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    # "blip" after changepoint - within gap limit
    d3 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                   3.0, 1.9, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    # "blip" after changepoint - outside gap limit
    d4 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                   3.0, 3.0, 3.0, 1.9, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    # "blip" before and after chanepoint - within gap limit
    d5 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.1, 2.0,
                   3.0, 1.9, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    # "blip" before and after changepoint - outside gap limit
    d6 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.1, 2.0, 2.0, 2.0, 2.0,
                   3.0, 3.0, 3.0, 1.9, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])

    # these types of noise can be handled
    # result should be identical to logical mask of ideal data
    for d in (d0, d1, d2, d4, d6):
        fd = make_mock_fd(force=f0, distance=d)
        sub = fd._sliced_by_distance(min_dist=min_dist, max_dist=max_dist, max_gap=max_gap)
        assert np.all(np.equal(ts_in_range, sub.d.timestamps))

    # these types of noise cannot be handled
    # result should be longer or shorter than expected
    for d in (d3, d5):
        fd = make_mock_fd(force=f0, distance=d)
        sub = fd._sliced_by_distance(min_dist=min_dist, max_dist=max_dist, max_gap=max_gap)
        with pytest.raises(ValueError) as exc:
            assert np.all(np.equal(ts_in_range, sub.d.timestamps))
        assert str(exc.value).strip() == ("operands could not be broadcast together with shapes "
                                          f"({ts_in_range.size},) ({sub.d.timestamps.size},)")


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
    np.testing.assert_allclose(getattr(fd1_sub, changed_var).data, getattr(fd1, changed_var).data - 1)
    np.testing.assert_allclose(getattr(fd1_sub, other_var).data, getattr(fd1, other_var).data)


def test_distance_offset_tracking_lost():
    """A value of 0 means the tracking was lost. This value should not change when subtracting baselines. An
    unfortunate side effect of using a regular number for this is that when subtraction leads to a distance of zero,
    this will lead to that data becoming a missing value point. This should not happen for real data though."""
    fd1 = make_mock_fd(force=[1, 2, 3], distance=[2, 0, 4], start=0)
    sub = fd1.with_offset(distance_offset=-1)
    np.testing.assert_allclose(sub.d.data, [1, 0, 3])


def test_subtract_too_much_distance():
    """Tests whether we get an exception when we subtract more than the lowest valid distance value"""
    fd1 = make_mock_fd(force=[1, 2, 3], distance=[2.0, 0.0, 4.0], start=0)

    np.testing.assert_allclose(fd1.with_offset(distance_offset=-1.0).d.data, [1.0, 0.0, 3.0])

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

    fdr = fd1._sliced(**slice_parameters)
    np.testing.assert_allclose(fdr.f, f)
    np.testing.assert_allclose(fdr.d, d)


def test_copy_behaviour_with_offset(h5_file):
    """Test whether with_offset is successful when there's an actual file attached. The reason this is tested explicitly
    is because with_offset may not deepcopy the handle it holds to the parent file."""
    fd1 = make_mock_fd(force=[1, 2, 3], distance=[2.0, 0.0, 4.0], start=0, file=h5_file)
    fd1.with_offset(distance_offset=-1.0)


def test_with_channels(fd_h5_file):
    fd_h5, _ = fd_h5_file
    f = File.from_h5py(fd_h5)
    fd = f.fdcurves["fd1"]
    assert fd._primary_force_channel == "2"
    assert fd._primary_distance_channel == "1"

    fd2 = fd.with_channels(force="1", distance="2")
    assert fd2._primary_force_channel == "1"
    assert fd2._primary_distance_channel == "2"


def test_with_baseline_correction(fd_h5_file):
    fd_h5, (timestamps_lf, true_force_lf) = fd_h5_file
    f = File.from_h5py(fd_h5)
    fd = f.fdcurves["fd1"]

    fd2 = fd.with_baseline_corrected_x()
    assert np.all(np.equal(fd.d.data, fd2.d.data))
    # compare to manually downsampled "true" force data
    assert np.all(np.equal(timestamps_lf, fd2.f.timestamps))
    assert np.all(np.equal(true_force_lf, fd2.f.data))
