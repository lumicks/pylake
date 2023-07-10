import pytest
import numpy as np
from lumicks.pylake.kymotracker.detail.denoising import generate_bspline_kernels


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
