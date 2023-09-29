import numpy as np
import pytest


def test_calibrate_to_kbp(test_kymo):
    kymo, ref = test_kymo
    length_kbp = 12.000
    n_pixels = ref.metadata.pixels_per_line

    kymo_bp = kymo.calibrate_to_kbp(length_kbp)

    # test that default calibration is in microns
    assert kymo._calibration.unit == "um"
    assert kymo._calibration.value == 0.1

    # test that calibration is stored as kilobase-pairs
    assert kymo_bp._calibration.unit == "kbp"
    np.testing.assert_allclose(kymo_bp._calibration.value, length_kbp / n_pixels)

    # test conversion from microns to calibration units
    np.testing.assert_allclose(
        kymo._calibration.value * n_pixels,
        ref.metadata.pixelsize_um[0] * n_pixels,
    )
    np.testing.assert_allclose(kymo.pixelsize, ref.metadata.pixelsize_um[0])
    np.testing.assert_allclose(kymo_bp._calibration.value * n_pixels, length_kbp)
    np.testing.assert_allclose(kymo_bp.pixelsize, length_kbp / n_pixels)


def check_factory_forwarding(kymo1, kymo2, check_timestamps):
    """test that all factories were forwarded from original instance"""
    assert kymo1._image_factory == kymo2._image_factory
    assert kymo1._timestamp_factory == kymo2._timestamp_factory
    assert kymo1._line_time_factory == kymo2._line_time_factory
    assert kymo1._pixelsize_factory == kymo2._pixelsize_factory
    assert kymo1._pixelcount_factory == kymo2._pixelcount_factory
    np.testing.assert_allclose(kymo1.get_image("red"), kymo2.get_image("red"))
    if check_timestamps:
        np.testing.assert_allclose(kymo1.timestamps, kymo2.timestamps)


def test_calibrate_sliced_cropped(test_kymo):
    kymo, ref = test_kymo
    length_kbp = 12.000
    n_pixels = ref.metadata.pixels_per_line

    kymo_bp = kymo.calibrate_to_kbp(length_kbp)

    # check that calibration is supported for any processing (downsampling/cropping)
    # and that data remains the same after calibration
    sliced_kymo = kymo["0s":"110s"]
    cropped_kymo = kymo.crop_by_distance(0, 0.4)
    n_cropped_pixels = cropped_kymo._num_pixels[0]

    sliced_kymo_bp = sliced_kymo.calibrate_to_kbp(length_kbp)
    cropped_kymo_bp = cropped_kymo.calibrate_to_kbp(length_kbp * (n_cropped_pixels / n_pixels))

    check_factory_forwarding(kymo, kymo_bp, True)
    check_factory_forwarding(sliced_kymo, sliced_kymo_bp, True)
    check_factory_forwarding(cropped_kymo, cropped_kymo_bp, True)

    # if properly calibrated, cropping should not change pixel size
    np.testing.assert_allclose(kymo_bp.pixelsize[0], cropped_kymo_bp.pixelsize[0])
    # but will change total length
    np.testing.assert_allclose(
        kymo_bp._calibration.value * n_cropped_pixels,
        cropped_kymo_bp._calibration.value * cropped_kymo_bp._num_pixels[0],
    )


def test_calibrate_with_downsampling(test_kymo):
    kymo, _ = test_kymo
    length_kbp = 12.000

    ds_kymo_time = kymo.downsampled_by(time_factor=2)
    ds_kymo_pos = kymo.downsampled_by(position_factor=3)
    ds_kymo_both = kymo.downsampled_by(time_factor=2, position_factor=3)

    ds_kymo_time_bp = ds_kymo_time.calibrate_to_kbp(length_kbp)
    ds_kymo_pos_bp = ds_kymo_pos.calibrate_to_kbp(length_kbp)  # total length does not change
    ds_kymo_both_bp = ds_kymo_both.calibrate_to_kbp(length_kbp)

    check_factory_forwarding(ds_kymo_time, ds_kymo_time_bp, False)
    check_factory_forwarding(ds_kymo_pos, ds_kymo_pos_bp, True)
    check_factory_forwarding(ds_kymo_both, ds_kymo_both_bp, False)


@pytest.mark.parametrize("crop", [False, True])
def test_flip_kymo(test_kymo, crop):
    kymo, _ = test_kymo
    if crop:
        kymo = kymo.crop_by_distance(0.01, 0.03)
    kymo_flipped = kymo.flip()
    for channel in ("red", "green", "blue", "rgb"):
        np.testing.assert_allclose(
            kymo_flipped.get_image(channel=channel),
            np.flip(kymo.get_image(channel=channel), axis=0),
        )
        np.testing.assert_allclose(
            kymo_flipped.flip().get_image(channel=channel),
            kymo.get_image(channel=channel),
        )
