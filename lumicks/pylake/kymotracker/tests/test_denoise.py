import pytest
import numpy as np
from lumicks.pylake.kymotracker.detail.denoising import *


def test_bspline_kernels_empty():
    assert not generate_bspline_kernels(0)


@pytest.mark.parametrize(
    "num_kernels, kernel_index, reference_kernel",
    [
        (2, 0, [1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16]),
        (2, 1, [1 / 16, 0, 1 / 4, 0, 3 / 8, 0, 1 / 4, 0, 1 / 16]),
        (3, 2, [1 / 16, 0, 0, 0, 1 / 4, 0, 0, 0, 3 / 8, 0, 0, 0, 1 / 4, 0, 0, 0, 1 / 16]),
    ],
)
def test_bspline_kernels(num_kernels, kernel_index, reference_kernel):
    kernels = generate_bspline_kernels(num_kernels)
    np.testing.assert_allclose(kernels[kernel_index].T, [reference_kernel])
    assert len(kernels) == num_kernels


def pad_kernels():
    eq_kernels = equal_length([np.array([1]), np.array([1, 2, 3]), np.array([1, 2, 3, 4, 5])])
    np.testing.assert_allclose(eq_kernels[0], [0, 0, 1, 0, 0])
    np.testing.assert_allclose(eq_kernels[1], [0, 1, 2, 3, 0])
    np.testing.assert_allclose(eq_kernels[2], [1, 2, 3, 4, 5])

    for bad in (
        [],
        [[1, 2]],
        [[1], [1, 2, 3], [1, 2, 3, 4]],
    ):
        with pytest.raises(
            ValueError, match=r"This function should only be used with odd sized kernels"
        ):
            equal_length(bad)


def test_product_filters():
    prod_kernels = generate_product_filters(
        [np.array([[1, 3, 5]]), np.array([[1, 2, 3]]), np.array([[1, 3, 5]])]
    )
    np.testing.assert_allclose(prod_kernels[0], [[1]])
    np.testing.assert_allclose(prod_kernels[1], [[1, 3, 5]])
    np.testing.assert_allclose(prod_kernels[2], [[1, 5, 14, 19, 15]])
    np.testing.assert_allclose(prod_kernels[3], [[1, 8, 34, 86, 142, 140, 75]])
