import re

import numpy as np
import pytest
import matplotlib as mpl
from numpy.testing import assert_array_equal

from lumicks.pylake.detail.confocal import timestamp_mean
from lumicks.pylake.detail.utilities import *
from lumicks.pylake.detail.utilities import (
    method_cache,
    will_mul_overflow,
    could_sum_overflow,
    replace_key_aliases,
)


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
    assert not will_mul_overflow(2, np.int64(2**62 - 1))
    assert will_mul_overflow(2, np.int64(2**62))
    assert will_mul_overflow(2, np.int64(2**63 - 1))

    assert not will_mul_overflow(np.array(2), np.array(2**62 - 1, dtype=np.int64))
    assert will_mul_overflow(np.array(2), np.array(2**62, dtype=np.int64))
    assert will_mul_overflow(np.array(2), np.array(2**63 - 1, dtype=np.int64))


def test_could_sum_overflow():
    assert not could_sum_overflow(np.array([1, 2**62 - 1]))
    assert could_sum_overflow(np.array([1, 2**62]))
    assert could_sum_overflow(np.array([1, 2**63 - 1]))

    assert not could_sum_overflow(np.array([1, 1, 2**61]))
    assert could_sum_overflow(np.array([1, 1, 2**62]))
    assert could_sum_overflow(np.array([1, 1, 2**63 - 1]))

    assert not could_sum_overflow(np.array([[1, 1], [1, 1]]), axis=0)
    assert not could_sum_overflow(np.array([[1, 1], [1, 1]]), axis=1)
    assert not could_sum_overflow(np.array([[1, 2**62 - 1], [1, 2**62 - 1]]), axis=0)
    assert not could_sum_overflow(np.array([[1, 2**62 - 1], [1, 2**62 - 1]]), axis=1)
    assert could_sum_overflow(np.array([[1, 2**62], [1, 2**62]]), axis=0)
    assert could_sum_overflow(np.array([[1, 2**62], [1, 2**62]]), axis=1)


def assert_1d(a):
    assert timestamp_mean(np.array(a)) == sum(a) // len(a)


def test_timestamp_mean():
    n = 2**62
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
    n = 2**62 // 4
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


@pytest.mark.parametrize(
    "src_dict, aliases, result_dict, valid",
    [
        ({}, [], {}, True),
        ({"test": 5}, [[]], {"test": 5}, True),
        ({"test": 5}, [["test"]], {"test": 5}, True),
        ({}, [["test", "test2"]], {}, True),
        ({"test": 5}, [["test", "yes"]], {"test": 5}, True),
        ({"test": 5}, [["yes", "test"]], {"yes": 5}, True),
        ({"test": 5}, [["yes", "yup", "test"]], {"yes": 5}, True),
        ({"test": 5, "yes": 5}, [["yes", "yup", "test"]], {}, False),  # Duplicate is invalid
        (
            {"test": 5, "second": 6},
            [["yes", "test"], ["good", "second"]],
            {"yes": 5, "good": 6},
            True,
        ),
        (
            {"test": 5, "second": 6},
            [["yes", "no"], ["good", "second"]],
            {"test": 5, "good": 6},
            True,
        ),
    ],
)
def test_key_aliases(src_dict, aliases, result_dict, valid):
    if valid:
        assert replace_key_aliases(src_dict, *aliases) == result_dict
    else:
        with pytest.raises(ValueError):
            replace_key_aliases(src_dict, *aliases)


@pytest.mark.parametrize("tst", [1, 2])
def test_freezing(reference_data, tst):
    test_data = np.array([[1, 2, 3], [1, 2, 5]])
    np.testing.assert_allclose(test_data, reference_data(test_data))
    test_data = np.array([[1, 2, 3], [1, 2, 6]])
    np.testing.assert_allclose(test_data, reference_data(test_data, test_name="mytest"))
    test_data = np.array([[1, 2, 3], [1, 2, 7]])
    np.testing.assert_allclose(
        test_data, reference_data(test_data, file_name=f"forced_filename_{tst}")
    )
    ref_dict = {"a": 5, "b": np.array([1, tst, 3])}
    test_dict = reference_data(ref_dict, test_name="dict")
    assert test_dict["a"] == ref_dict["a"]
    np.testing.assert_allclose(test_dict["b"], ref_dict["b"])

    test_data = [[1, 2], [1, 2, 3]]
    ref_data = reference_data(test_data, test_name="ragged")
    for test, ref in zip(test_data, ref_data):
        np.testing.assert_allclose(test, ref)

    test_data = [[1, 2, 3], [1, 2, 3]]
    np.testing.assert_allclose(test_data, reference_data(test_data, test_name="non_ndarray_matrix"))


@pytest.mark.parametrize("tst", [1, 2])
def test_ref_dict_freezing(compare_to_reference_dict, reference_data, tst):
    ref_dict = {"a": 5, "b": np.pi if tst == 1 else 1e-12}
    test_dict = reference_data(ref_dict, test_name="dict", json=True)
    np.testing.assert_allclose(list(test_dict.values()), list(ref_dict.values()))
    compare_to_reference_dict(test_dict)


def test_ref_dict_freezing_fail(request, compare_to_reference_dict):
    if request.config.getoption("--update_reference_data"):
        # Don't rewrite these as they intentionally fail
        return

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Differences with reference data detected.\n"
            "a: 5 vs 5 (match)\n"
            "b: 2 vs 3 (difference)"
        ),
    ):
        compare_to_reference_dict({"a": 5, "b": 2})

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Differences with reference data detected.\n"
            "a: 5 vs 5 (match)\n"
            "b: missing vs 3 (reference only)"
        ),
    ):
        compare_to_reference_dict({"a": 5})

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Differences with reference data detected.\n"
            "a: 5 vs 5 (match)\n"
            "b: 3 vs missing (test only)"
        ),
    ):
        compare_to_reference_dict({"a": 5, "b": 3}, file_name="ref_dict_freezing_2")


def test_cache_method():
    calls = 0

    def call():
        nonlocal calls
        calls += 1

    class Test:
        def __init__(self):
            self._cache = {}

        @property
        @method_cache("example_property")
        def example_property(self):
            call()
            return 10

        @method_cache("example_method")
        def example_method(self, argument=5):
            call()
            return argument

    test = Test()
    assert len(test._cache) == 0
    assert test.example_property == 10
    assert len(test._cache) == 1
    assert calls == 1
    assert test.example_property == 10
    assert calls == 1
    assert len(test._cache) == 1

    assert test.example_method() == 5
    assert calls == 2
    assert len(test._cache) == 2
    assert test.example_method() == 5
    assert calls == 2
    assert len(test._cache) == 2
    assert test.example_method(6) == 6
    assert calls == 3
    assert len(test._cache) == 3
    assert test.example_method(6) == 6
    assert calls == 3
    assert len(test._cache) == 3
