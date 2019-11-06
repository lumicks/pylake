from lumicks.pylake.detail.utilities import first, unique, unique_idx
import pytest


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


def test_unique_idx():
    uiq, inv = unique_idx(['str', 'str', 'hmm', 'potato', 'hmm', 'str'])
    assert(uiq == ['str', 'hmm', 'potato'])
    assert(inv == [0, 0, 1, 2, 1, 0])