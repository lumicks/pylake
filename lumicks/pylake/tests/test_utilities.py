from lumicks.pylake.detail.utilities import *
from lumicks.pylake.detail.confocal import timestamp_mean
from lumicks.pylake.detail.utilities import will_mul_overflow, could_sum_overflow
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
import numpy as np


def test_first():
    assert first((1, 2, 3), condition=lambda x: x % 2 == 0) == 2
    assert first(range(3, 100)) == 3

    with pytest.raises(StopIteration):
        first((1, 2, 3), condition=lambda x: x % 5 == 0)
    with pytest.raises(StopIteration):
        first(())


def test_unique():
    uiq = unique(["str", "str", "hmm", "potato", "hmm", "str"])
    assert uiq == ["str", "hmm", "potato"]


def test_colors():
    [mpl.colors.to_rgb(get_color(k)) for k in range(30)]
    np.testing.assert_allclose(lighten_color([0.5, 0, 0], 0.2), [0.7, 0, 0])


def test_find_contiguous():
    def check_blocks_are_true(mask, ranges):
        for rng in ranges:
            assert np.all(mask[slice(*rng)])

    data = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])

    mask = data
    ranges, lengths = find_contiguous(mask)
    assert np.all(np.equal(ranges, [[1, 10]]))
    assert np.all(np.equal(lengths, [9]))
    check_blocks_are_true(mask, ranges)

    mask = data < 10
    ranges, lengths = find_contiguous(mask)
    assert np.all(np.equal(ranges, [[0, 11]]))
    assert np.all(np.equal(lengths, [11]))
    check_blocks_are_true(mask, ranges)

    mask = data > 10
    ranges, lengths = find_contiguous(mask)
    assert len(ranges) == 0
    assert len(lengths) == 0
    check_blocks_are_true(mask, ranges)

    mask = data < 4
    ranges, lengths = find_contiguous(mask)
    assert np.all(np.equal(ranges, [[0, 4], [7, 11]]))
    assert np.all(np.equal(lengths, [4, 4]))
    check_blocks_are_true(mask, ranges)

    data = np.arange(10)

    mask = data <= 5
    ranges, lengths = find_contiguous(mask)
    assert np.all(np.equal(ranges, [[0, 6]]))
    assert np.all(np.equal(lengths, [6]))
    check_blocks_are_true(mask, ranges)

    mask = data >= 5
    ranges, lengths = find_contiguous(mask)
    assert np.all(np.equal(ranges, [[5, 10]]))
    assert np.all(np.equal(lengths, [5]))
    check_blocks_are_true(mask, ranges)


@pytest.mark.parametrize(
    "data,factor,avg,std",
    [
        [np.arange(10), 2, [0.5, 2.5, 4.5, 6.5, 8.5], [0.5, 0.5, 0.5, 0.5, 0.5]],
        [np.arange(0, 10, 2), 1, [0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        [np.arange(0, 10, 2), 2, [1.0, 5.0], [1.0, 1.0]],
        [np.arange(0, 11, 2), 2, [1.0, 5.0, 9.0], [1.0, 1.0, 1.0]],
    ],
)
def test_downsample(data, factor, avg, std):
    np.testing.assert_allclose(avg, downsample(data, factor, reduce=np.mean))
    np.testing.assert_allclose(std, downsample(data, factor, reduce=np.std))


def test_will_mul_overflow():
    assert not will_mul_overflow(2, np.int64(2 ** 62 - 1))
    assert will_mul_overflow(2, np.int64(2 ** 62))
    assert will_mul_overflow(2, np.int64(2 ** 63 - 1))

    assert not will_mul_overflow(np.array(2), np.array(2 ** 62 - 1, dtype=np.int64))
    assert will_mul_overflow(np.array(2), np.array(2 ** 62, dtype=np.int64))
    assert will_mul_overflow(np.array(2), np.array(2 ** 63 - 1, dtype=np.int64))


def test_could_sum_overflow():
    assert not could_sum_overflow(np.array([1, 2 ** 62 - 1]))
    assert could_sum_overflow(np.array([1, 2 ** 62]))
    assert could_sum_overflow(np.array([1, 2 ** 63 - 1]))

    assert not could_sum_overflow(np.array([1, 1, 2 ** 61]))
    assert could_sum_overflow(np.array([1, 1, 2 ** 62]))
    assert could_sum_overflow(np.array([1, 1, 2 ** 63 - 1]))

    assert not could_sum_overflow(np.array([[1, 1], [1, 1]]), axis=0)
    assert not could_sum_overflow(np.array([[1, 1], [1, 1]]), axis=1)
    assert not could_sum_overflow(np.array([[1, 2 ** 62 - 1], [1, 2 ** 62 - 1]]), axis=0)
    assert not could_sum_overflow(np.array([[1, 2 ** 62 - 1], [1, 2 ** 62 - 1]]), axis=1)
    assert could_sum_overflow(np.array([[1, 2 ** 62], [1, 2 ** 62]]), axis=0)
    assert could_sum_overflow(np.array([[1, 2 ** 62], [1, 2 ** 62]]), axis=1)


def assert_1d(a):
    assert timestamp_mean(np.array(a)) == sum(a) // len(a)


def test_timestamp_mean():
    n = 2 ** 62
    assert_1d([2, 4])
    assert_1d([n - 1, n - 3])
    assert_1d([0, 0, n - 1, n - 1])
    assert_1d([0, n - 1, 0, 0, 0, 0])
    assert_1d([n, n])

    assert_array_equal(timestamp_mean(np.array([[2, 4], [2, 4]]), axis=0), [2, 4])
    assert_array_equal(timestamp_mean(np.array([[2, 4], [2, 4]]), axis=1), [3, 3])
    assert_array_equal(timestamp_mean(np.array([[n - 1, n - 3]] * 2), axis=0), [n - 1, n - 3])
    assert_array_equal(timestamp_mean(np.array([[n - 1, n - 3]] * 2), axis=1), [n - 2, n - 2])

    assert_array_equal(
        timestamp_mean(np.array([[n - 1, n - 3, n - 5]] * 2), axis=0), [n - 1, n - 3, n - 5]
    )
    assert_array_equal(
        timestamp_mean(np.array([[n - 1, n - 3, n - 5, n - 7]] * 2), axis=1), [n - 4, n - 4]
    )


def test_timestamp_mean_2d():
    """Test 2D behaviour of timestamp_mean."""
    n = 2 ** 62 // 4
    t_range = np.arange(0, n * 8, n, dtype=np.int64)
    ts = np.tile(t_range, (6, 1))

    # Note that small round-off errors still occur since we average blocks. Note that these errors
    # are far smaller than with FP, and will only occur for cases that are otherwise extremely rare.
    np.testing.assert_equal(timestamp_mean(ts, 0) - t_range, [0, -4, -2, 0, -4, -2, 0, -4])
    np.testing.assert_equal(timestamp_mean(ts.T, 1) - t_range, [0, -4, -2, 0, -4, -2, 0, -4])

    np.testing.assert_equal(timestamp_mean(ts, 1), np.tile(n * 7 // 2, (6,)))
    np.testing.assert_equal(timestamp_mean(ts.T, 0), np.tile(n * 7 // 2, (6,)))


def test_docstring_wrapper():
    def func1():
        """This one has a docstring"""

    @use_docstring_from(func1)
    def func2():
        """This one should use the other one's docstring"""

    assert func1.__doc__ == func2.__doc__
