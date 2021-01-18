from lumicks.pylake.detail.utilities import first, unique, get_color, lighten_color, find_contiguous
import pytest
import matplotlib as mpl
import numpy as np


def test_first():
    assert(first((1, 2, 3), condition=lambda x: x % 2 == 0) == 2)
    assert(first(range(3, 100)) == 3)

    with pytest.raises(StopIteration):
        first((1, 2, 3), condition=lambda x: x % 5 == 0)
    with pytest.raises(StopIteration):
        first(())


def test_unique():
    uiq = unique(['str', 'str', 'hmm', 'potato', 'hmm', 'str'])
    assert(uiq == ['str', 'hmm', 'potato'])


def test_colors():
    [mpl.colors.to_rgb(get_color(k)) for k in range(30)]
    assert np.allclose(lighten_color([0.5, 0, 0], .2), [.7, 0, 0])


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
    assert np.all(np.equal(ranges, [[0, 4],
                                    [7, 11]]))
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
