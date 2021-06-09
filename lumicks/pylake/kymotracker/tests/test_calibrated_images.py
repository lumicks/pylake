import pytest
import numpy as np
from lumicks.pylake.kymotracker.detail.calibrated_images import CalibratedKymographChannel


@pytest.fixture(scope="module")
def test_img():
    return np.array(
        [
            [0, 12, 0, 12, 0],
            [0, 0, 0, 0, 0],
            [12, 0, 0, 0, 12],
            [0, 12, 12, 12, 0],
        ],
        dtype=np.uint8,
    )


def test_fields_calibrated_kymograph_channel(test_img):
    calibrated_channel = CalibratedKymographChannel(
        "test", test_img, time_step_ns=3e9, pixel_size=7
    )
    assert calibrated_channel.name == "test"
    assert np.all(calibrated_channel.data == test_img)
    np.testing.assert_allclose(calibrated_channel.time_step_ns, 3e9)
    np.testing.assert_allclose(calibrated_channel._pixel_size, 7)
    np.testing.assert_allclose(calibrated_channel.to_position(np.array([1, 2, 3])), [7, 14, 21])
    np.testing.assert_allclose(calibrated_channel.to_seconds(np.array([1, 2, 3])), [3, 6, 9])


@pytest.mark.parametrize(
    "rect,valid_result",
    [
        ([[4, 6], [14, 15]], [[12, 0, 12], [0, 0, 0]]),
        ([[7, 15], [14, 50]], [[0, 0], [12, 12]]),
    ],
)
def test_in_rect(test_img, rect, valid_result):
    calibrated_channel = CalibratedKymographChannel(
        "test", test_img, time_step_ns=3e9, pixel_size=7
    )
    np.testing.assert_allclose(calibrated_channel.get_rect(rect), valid_result)


@pytest.mark.parametrize(
    "rect",
    [
        ((10, 10), (5, 20)),
        ((5, 20), (10, 10)),
        ((5, 16), (10, 18)),
        ((26, 5), (28, 7)),
        ((-1, 8), (22, 20)),
        ((5, -1), (22, 20)),
        ((5, 8), (-1, 20)),
        ((5, 8), (22, -1))
    ],
)
def test_rect_errors(rect):
    data = np.random.rand(15, 25)
    calibrated_channel = CalibratedKymographChannel("test", data, time_step_ns=1e9, pixel_size=1)
    np.testing.assert_allclose(calibrated_channel.get_rect(((5, 8), (22, 20))), data[8:20, 5:22])

    with pytest.raises(IndexError):
        calibrated_channel.get_rect(rect)
