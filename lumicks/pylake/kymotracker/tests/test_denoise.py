import pytest
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


@pytest.mark.parametrize(
    "kernel, coefficients",
    [
        ([1.0], (1.0, 3 / 8)),  # Anscombe transform's c
        ([1.0, 0.0, 1.0], (1 / np.sqrt(2), 3 / 8)),
        ([1.0, 2.0, 3.0], (0.4082482904638631, 0.7559523809523807)),
        ([1.0, 3.0, 3.0, 7.0], (0.2672612419124244, 1.323529411764706)),
    ],
)
def test_variance_stabilizing_coefficients(kernel, coefficients):
    np.testing.assert_allclose(calculate_vst_coefficients(np.array(kernel)), coefficients)


@pytest.mark.parametrize(
    "kernel, prev_kernel, stdev",
    [
        ([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], 0.6123724356957945),
        ([1.0, 1.0, 2.0, 1.0, 2.0], [1.0, 0.0, 1.0, 0.0, 2.0], 0.15771001547014013),
        ([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], 0.25),
    ],
)
def test_variance_stabilizing_stdev(kernel, prev_kernel, stdev):
    np.testing.assert_allclose(
        calculate_vst_stdev(np.array(kernel), np.array(prev_kernel)),
        stdev,
    )


@pytest.mark.parametrize(
    "kernel, prev_kernel",
    [
        ([1.0, 0.0, 1.0], [0.0, 1.0]),
        ([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], [1.0, 0.0]),
    ],
)
def test_unequal_shape_vst_stdev(kernel, prev_kernel):
    with pytest.raises(
        ValueError, match=r"Kernel shapes must be equal to calculate standard deviation"
    ):
        calculate_vst_stdev(np.array(kernel), np.array(prev_kernel))


def test_variance_stabilizing_transform():
    img = np.array([[1, -1, 2, -2], [1, 2, 3, 4]])
    coeffs = (2, 3 / 8)
    ref = [
        [2.34520788, -1.58113883, 3.082207, -2.54950976],
        [2.34520788, 3.082207, 3.67423461, 4.18330013],
    ]
    result = variance_stabilizing_transform(img, coeffs)
    np.testing.assert_allclose(result, ref)
    np.testing.assert_allclose(inverse_variance_stabilizing_transform(result, coeffs), img)


def test_wavelets():
    image = [
        [0, 2, 0, 1, 0, 2, 0],
        [0, 0, 2, 0, 2, 0, 0],
        [0, 2, 0, 1, 0, 2, 0],
    ]

    ms_vst = MultiScaleVarianceStabilizingTransform(generate_bspline_kernels(3))
    coeffs = []
    for stabilized in (False, True):
        detail_coeffs, remainder = ms_vst._calculate_wavelets(np.asarray(image), stabilized)
        np.testing.assert_allclose(
            ms_vst._reconstruct_image(detail_coeffs, remainder, stabilized), image, atol=1e-6
        )
        coeffs.append(detail_coeffs)

    # Verify that stabilization results in different detail coefficients
    for c_not_stabilized, c_stabilized in zip(*coeffs):
        assert np.max(np.abs(c_not_stabilized - c_stabilized)) > 1e-2
