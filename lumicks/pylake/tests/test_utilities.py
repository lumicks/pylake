from lumicks.pylake.detail.utilities import first, unique, unique_idx, clamp_step
import pytest
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


def test_unique_idx():
    uiq, inv = unique_idx(['str', 'str', 'hmm', 'potato', 'hmm', 'str'])
    assert(uiq == ['str', 'hmm', 'potato'])
    assert(inv == [0, 0, 1, 2, 1, 0])

def test_clamp_vector():
    # Positive quadrant
    assert np.allclose(clamp_step(np.array([1, 1]), np.array([-2, -4]), np.array([0, 0]), np.array([2, 2]))[0],
                       np.array([0.5, 0.0]))
    assert np.allclose(clamp_step(np.array([1, 1]), np.array([-4, -4]), np.array([0, 0]), np.array([2, 2]))[0],
                       np.array([0, 0]))
    assert np.allclose(clamp_step(np.array([1, 1]), np.array([4, 4]), np.array([0, 0]), np.array([2, 2]))[0],
                       np.array([2, 2]))
    assert np.allclose(clamp_step(np.array([1, 1]), np.array([2, 4]), np.array([0, 0]), np.array([2, 2]))[0],
                       np.array([1.5, 2]))

    assert np.allclose(clamp_step(np.array([1, 1]), np.array([-2, 4]), np.array([0, 0]), np.array([2, 2]))[0],
                       np.array([0.5, 2]))
    assert np.allclose(clamp_step(np.array([1, 1]), np.array([2, -4]), np.array([0, 0]), np.array([2, 2]))[0],
                       np.array([1.5, 0]))

    assert np.allclose(clamp_step(np.array([1, 1]), np.array([0, .5]), np.array([0, 0]), np.array([2, 2]))[0],
                       np.array([1.0, 1.5]))
    assert np.allclose(clamp_step(np.array([1, 1]), np.array([.5, 0]), np.array([0, 0]), np.array([2, 2]))[0],
                       np.array([1.5, 1]))
    assert np.allclose(clamp_step(np.array([1, 1]), np.array([-.5, 0]), np.array([0, 0]), np.array([2, 2]))[0],
                       np.array([0.5, 1]))
    assert np.allclose(clamp_step(np.array([1, 1]), np.array([-.5, -.5]), np.array([0, 0]), np.array([2, 2]))[0],
                       np.array([0.5, 0.5]))
    assert np.allclose(clamp_step(np.array([3, 3]), np.array([-.5, 0.0]), np.array([2, 2]), np.array([4, 4]))[0],
                       np.array([2.5, 3.0]))
    assert np.allclose(clamp_step(np.array([3, 3]), np.array([-10.0, 0.0]), np.array([2, 2]), np.array([4, 4]))[0],
                       np.array([2.0, 3.0]))

    # Negative quadrant
    assert np.allclose(clamp_step(np.array([-1, -1]), np.array([-2, -4]), np.array([-2, -2]), np.array([0, 0]))[0],
                       np.array([-1.5, -2]))
    assert np.allclose(clamp_step(np.array([-1, -1]), np.array([-4, -4]), np.array([-2, -2]), np.array([0, 0]))[0],
                       np.array([-2, -2]))
    assert np.allclose(clamp_step(np.array([-1, -1]), np.array([4, 4]), np.array([-2, -2]), np.array([0, 0]))[0],
                       np.array([0, 0]))
    assert np.allclose(clamp_step(np.array([-1, -1]), np.array([2, 4]), np.array([-2, -2]), np.array([0, 0]))[0],
                       np.array([-0.5, 0]))

    assert np.allclose(clamp_step(np.array([-1, -1]), np.array([-2, 4]), np.array([-2, -2]), np.array([0, 0]))[0],
                       np.array([-1.5, 0.0]))
    assert np.allclose(clamp_step(np.array([-1, -1]), np.array([2, -4]), np.array([-2, -2]), np.array([0, 0]))[0],
                       np.array([-0.5, -2]))

    # Both quadrants
    assert np.allclose(clamp_step(np.array([-1, -1]), np.array([-2, -4]), np.array([-2, -2]), np.array([2, 2]))[0],
                       np.array([-1.5, -2]))
    assert np.allclose(clamp_step(np.array([-1, -1]), np.array([-4, -4]), np.array([-2, -2]), np.array([2, 2]))[0],
                       np.array([-2, -2]))
    assert np.allclose(clamp_step(np.array([-1, -1]), np.array([4, 4]), np.array([-2, -2]), np.array([2, 2]))[0],
                       np.array([2, 2]))
    assert np.allclose(clamp_step(np.array([-1, -1]), np.array([2, 4]), np.array([-2, -2]), np.array([2, 2]))[0],
                       np.array([-1.0+2.0*(3.0/4.0), 2]))

    assert np.allclose(clamp_step(np.array([-1, -1]), np.array([-2, 4]), np.array([-2, -2]), np.array([2, 2]))[0],
                       np.array([-2.0, 1.0]))
    assert np.allclose(clamp_step(np.array([-1, -1]), np.array([2, -4]), np.array([-2, -2]), np.array([2, 2]))[0],
                       np.array([-0.5, -2.0]))