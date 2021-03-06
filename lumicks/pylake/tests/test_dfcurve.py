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
