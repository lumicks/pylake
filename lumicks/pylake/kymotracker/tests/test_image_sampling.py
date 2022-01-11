import numpy as np
from lumicks.pylake.kymotracker.detail.calibrated_images import CalibratedKymographChannel
from lumicks.pylake.kymotracker.kymoline import KymoLine


def test_sampling():
    test_data = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    test_img = CalibratedKymographChannel("test", test_data, 10e9, 5)

    # Tests the bound handling
    kymoline = KymoLine([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], test_img)
    np.testing.assert_allclose(kymoline.sample_from_image(50), [0, 2, 3, 2, 0])
    np.testing.assert_allclose(kymoline.sample_from_image(2), [0, 2, 3, 2, 0])
    np.testing.assert_allclose(kymoline.sample_from_image(1), [0, 2, 2, 2, 0])
    np.testing.assert_allclose(kymoline.sample_from_image(0), [0, 1, 1, 1, 0])
    np.testing.assert_allclose(
        KymoLine([0, 1, 2, 3, 4], [4, 4, 4, 4, 4], test_img).sample_from_image(0), [0, 0, 1, 1, 0]
    )

    kymoline = KymoLine([0.1, 1.1, 2.1, 3.1, 4.1], [0.1, 1.1, 2.1, 3.1, 4.1], test_img)
    np.testing.assert_allclose(kymoline.sample_from_image(50), [0, 2, 3, 2, 0])
    np.testing.assert_allclose(kymoline.sample_from_image(2), [0, 2, 3, 2, 0])
    np.testing.assert_allclose(kymoline.sample_from_image(1), [0, 2, 2, 2, 0])
    np.testing.assert_allclose(kymoline.sample_from_image(0), [0, 1, 1, 1, 0])
    kymoline = KymoLine([0.1, 1.1, 2.1, 3.1, 4.1], [4.1, 4.1, 4.1, 4.1, 4.1], test_img)
    np.testing.assert_allclose(kymoline.sample_from_image(0), [0, 0, 1, 1, 0])


def test_kymoline_regression_sample_from_image_clamp():
    """This tests for a regression that occurred in sample_from_image. When sampling the image, we
    sample pixels in a region around the line. This sampling procedure is constrained to stay within
    the image. Previously, we used the incorrect axis to clamp the coordinate.
    """
    # Sampling the bottom row of a three pixel tall image will return [0, 0] instead of [1, 3];
    # since both coordinates would be clamped to the edge of the image (sampling nothing)."""
    img = CalibratedKymographChannel("test_data", np.array([[1, 1, 1], [3, 3, 3]]).T, 1e9, 1)
    assert np.array_equal(KymoLine([0, 1], [2, 2], img).sample_from_image(0), [1, 3])
