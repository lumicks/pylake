import numpy as np
from lumicks.pylake.kymotracker.kymotrack import KymoTrack
from lumicks.pylake.tests.data.mock_confocal import generate_kymo


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
    test_img = generate_kymo(
        "",
        test_data,
        pixel_size_nm=5000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0
    )

    # Tests the bound handling
    kymotrack = KymoTrack([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], test_img, "red")
    np.testing.assert_allclose(kymotrack.sample_from_image(50), [0, 2, 3, 2, 0])
    np.testing.assert_allclose(kymotrack.sample_from_image(2), [0, 2, 3, 2, 0])
    np.testing.assert_allclose(kymotrack.sample_from_image(1), [0, 2, 2, 2, 0])
    np.testing.assert_allclose(kymotrack.sample_from_image(0), [0, 1, 1, 1, 0])
    np.testing.assert_allclose(
        KymoTrack([0, 1, 2, 3, 4], [4, 4, 4, 4, 4], test_img, "red").sample_from_image(0), [0, 0, 1, 1, 0]
    )

    kymotrack = KymoTrack([0.1, 1.1, 2.1, 3.1, 4.1], [0.1, 1.1, 2.1, 3.1, 4.1], test_img, "red")
    np.testing.assert_allclose(kymotrack.sample_from_image(50), [0, 2, 3, 2, 0])
    np.testing.assert_allclose(kymotrack.sample_from_image(2), [0, 2, 3, 2, 0])
    np.testing.assert_allclose(kymotrack.sample_from_image(1), [0, 2, 2, 2, 0])
    np.testing.assert_allclose(kymotrack.sample_from_image(0), [0, 1, 1, 1, 0])
    kymotrack = KymoTrack([0.1, 1.1, 2.1, 3.1, 4.1], [4.1, 4.1, 4.1, 4.1, 4.1], test_img, "red")
    np.testing.assert_allclose(kymotrack.sample_from_image(0), [0, 0, 1, 1, 0])


def test_kymotrack_regression_sample_from_image_clamp():
    """This tests for a regression that occurred in sample_from_image. When sampling the image, we
    sample pixels in a region around the track. This sampling procedure is constrained to stay within
    the image. Previously, we used the incorrect axis to clamp the coordinate.
    """
    # Sampling the bottom row of a three pixel tall image will return [0, 0] instead of [1, 3];
    # since both coordinates would be clamped to the edge of the image (sampling nothing)."""

    img = generate_kymo(
        "",
        np.array([[1, 1, 1], [3, 3, 3]]).T,
        pixel_size_nm=1000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0
    )
    assert np.array_equal(KymoTrack([0, 1], [2, 2], img, "red").sample_from_image(0), [1, 3])
