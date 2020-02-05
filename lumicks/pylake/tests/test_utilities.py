from lumicks.pylake.detail.utilities import first, unique, get_color, lighten_color
import pytest
from matplotlib import colors
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
    [colors.to_rgb(get_color(k)) for k in range(30)]
    assert np.allclose(lighten_color([0.5, 0, 0], .2), [.7, 0, 0])
