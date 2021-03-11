import pytest
import numpy as np
from lumicks.pylake.detail.calibrated_images import CalibratedKymographChannel
from matplotlib.testing.decorators import cleanup
from lumicks import pylake


@pytest.fixture(scope="module")
def test_img():
    return np.array(
        [
            np.uint8([0, 12, 0, 12, 0]),
            np.uint8([0, 0, 0, 0, 0]),
            np.uint8([12, 0, 0, 0, 12]),
            np.uint8([0, 12, 12, 12, 0]),
        ]
    )


def test_fields_calibrated_kymograph_channel(test_img):
    calibrated_channel = CalibratedKymographChannel(
        "test", test_img, start=4e9, time_step=3e9, calibration=7
    )
    assert calibrated_channel.name == "test"
    assert np.allclose(calibrated_channel.start, 4e9)
    assert np.allclose(calibrated_channel.data, test_img)
    assert np.allclose(calibrated_channel.downsampling_factor, 1)
    assert np.allclose(calibrated_channel.time_step, 3e9)
    assert np.allclose(calibrated_channel._calibration, 7)
    assert np.allclose(calibrated_channel.to_coord(np.array([1, 2, 3])), [7, 14, 21])
    assert np.allclose(calibrated_channel.to_seconds(np.array([1, 2, 3])), [3, 6, 9])


def test_downsampling_calibrated_kymograph_channel(test_img):
    ds = np.array(
        [
            np.uint8([12, 12]),
            np.uint8([0, 0]),
            np.uint8([12, 0]),
            np.uint8([12, 24]),
        ]
    )
    calibrated_channel = CalibratedKymographChannel(
        "test", test_img, start=5e9, time_step=3e9, calibration=7
    ).downsampled_by(2)
    assert calibrated_channel.name == "test"
    assert np.allclose(calibrated_channel.data, ds)
    assert np.allclose(calibrated_channel.start, 5e9)
    assert np.allclose(calibrated_channel.downsampling_factor, 2)
    assert np.allclose(calibrated_channel.time_step, 3e9 * 2)
    assert np.allclose(calibrated_channel._calibration, 7)
    assert np.allclose(calibrated_channel.to_coord(np.array([1, 2, 3])), [7, 14, 21])
    assert np.allclose(calibrated_channel.to_seconds(np.array([1, 2, 3])), [6, 12, 18])


@cleanup
def test_plotting(test_img, h5_file):
    calibrated_channel = CalibratedKymographChannel(
        "test", test_img, start=5, time_step=3, calibration=7
    )
    calibrated_channel.plot()
    calibrated_channel.downsampled_by(2).plot()

    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        with pytest.raises(RuntimeError):
            f.kymos["Kymo1"].get_channel("red").plot()


@pytest.mark.parametrize("rect,valid_result", [
    ([[4, 6], [14, 15]], [[12, 0, 12], [0, 0, 0]]),
    ([[7, 15], [14, 50]], [[0, 0], [12, 12]]),
])
def test_in_rect(test_img, rect, valid_result):
    calibrated_channel = CalibratedKymographChannel(
        "test", test_img, start=5e9, time_step=3e9, calibration=7
    )
    np.allclose(calibrated_channel._get_rect(rect), valid_result)


def test_rect_errors():
    data = np.random.rand(15, 25)
    calibrated_channel = CalibratedKymographChannel(
        "test", data, start=5e9, time_step=1e9, calibration=1
    )
    assert(np.allclose(calibrated_channel._get_rect(((5, 8), (22, 20))), data[8:20, 5:22]))

    with pytest.raises(IndexError):
        calibrated_channel._get_rect(((10, 10), (5, 20)))

    with pytest.raises(IndexError):
        calibrated_channel._get_rect(((5, 20), (10, 10)))

    with pytest.raises(IndexError):
        calibrated_channel._get_rect(((5, 16), (10, 18)))

    with pytest.raises(IndexError):
        calibrated_channel._get_rect(((26, 5), (28, 7)))
