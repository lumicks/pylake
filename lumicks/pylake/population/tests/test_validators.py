import numpy as np

from lumicks.pylake.population.detail.validators import col, row


def test_row():
    assert row(1).shape == (1, 1)
    assert row([1, 2, 3]).shape == (1, 3)
    assert row(np.array([1, 2, 3])).shape == (1, 3)


def test_col():
    assert col(1).shape == (1, 1)
    assert col([1, 2, 3]).shape == (3, 1)
    assert col(np.array([1, 2, 3])).shape == (3, 1)
