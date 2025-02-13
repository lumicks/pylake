import numpy as np
import pytest

from lumicks.pylake.kymo import PositionUnit, PositionCalibration


def test_calibrate_to_kbp(test_kymo):
    kymo, ref = test_kymo
    length_kbp = 12.000
    n_pixels = ref.metadata.pixels_per_line

    kymo_bp = kymo.calibrate_to_kbp(length_kbp)

    # test that default calibration is in microns
    assert kymo._calibration.unit == PositionUnit.um
    assert kymo._calibration.pixelsize == 0.1

    # test that calibration is stored as kilobase-pairs
    assert kymo_bp._calibration.unit == PositionUnit.kbp
    np.testing.assert_allclose(kymo_bp._calibration.pixelsize, length_kbp / n_pixels)

    # test conversion from microns to calibration units
    np.testing.assert_allclose(
        kymo._calibration.pixelsize * n_pixels,
        ref.metadata.pixelsize_um[0] * n_pixels,
    )
    np.testing.assert_allclose(kymo.pixelsize, ref.metadata.pixelsize_um[0])
    np.testing.assert_allclose(kymo_bp._calibration.pixelsize * n_pixels, length_kbp)
    np.testing.assert_allclose(kymo_bp.pixelsize, length_kbp / n_pixels)

    start = 0.12
    end = 0.33
    n_pixels_tether = (end - start) / ref.metadata.pixelsize_um[0]

    kymo_bp = kymo.calibrate_to_kbp(length_kbp, start=start, end=end)
    np.testing.assert_allclose(kymo_bp._calibration.pixelsize * n_pixels_tether, length_kbp)
    np.testing.assert_allclose(kymo_bp.pixelsize, length_kbp / n_pixels_tether)

    with pytest.raises(RuntimeError, match="kymo is already calibrated in base pairs."):
        kymo_bp.calibrate_to_kbp(10)


def test_calibrate_to_kbp_invalid(test_kymo):
    with pytest.raises(
        ValueError, match="Start must be supplied with end or length_um."
    ):
        test_kymo[0].calibrate_to_kbp(1, start=None, end=0.33)
    with pytest.raises(
        ValueError, match="Start must be supplied with end or length_um."
    ):
        test_kymo[0].calibrate_to_kbp(1, start=None, length_um=10.0)
    with pytest.raises(
        ValueError, match="Either end or length_um must be supplied with start."
    ):
        test_kymo[0].calibrate_to_kbp(1, start=0.33, end=None)
    with pytest.raises(
        ValueError, match="Either end or length_um must be supplied with start."
    ):
        test_kymo[0].calibrate_to_kbp(1, start=0.33, length_um=None)
    with pytest.raises(
        ValueError, match="Cannot use landmarks with start."
    ):
        test_kymo[0].calibrate_to_kbp(1, start=0.33, landmarks=([1, 2], [3, 4]))
    with pytest.raises(
        ValueError, match="At least 2 landmarks must be supplied."
    ):
        test_kymo[0].calibrate_to_kbp(1, landmarks=([1, 2], ))


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
        kymo_bp._calibration.pixelsize * n_cropped_pixels,
        cropped_kymo_bp._calibration.pixelsize * cropped_kymo_bp._num_pixels[0],
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


def test_position_unit():
    assert PositionUnit.um.label == r"μm"
    assert PositionUnit.kbp.label == "kbp"
    assert PositionUnit.pixel.label == "pixels"

    assert str(PositionUnit.um) == "um"
    assert str(PositionUnit.kbp) == "kbp"
    assert str(PositionUnit.pixel) == "pixel"

    assert {PositionUnit.um, PositionUnit.um, PositionUnit.kbp} == {
        PositionUnit.um,
        PositionUnit.kbp,
    }


def test_enum_in_calibration():
    with pytest.raises(TypeError, match="`unit` must be a PositionUnit instance"):
        PositionCalibration("kbp", scale=0.42)

    c = PositionCalibration(PositionUnit.um, scale=0.42)
    assert c.unit.label == PositionUnit.um.label


def test_coordinate_transforms():
    px_coord = [0, 1.2, 3.14, 85]

    c = PositionCalibration(PositionUnit.kbp, scale=0.42)
    kbp_coord = [0, 0.504, 1.3188, 35.7]
    transformed = c.from_pixels(px_coord)
    np.testing.assert_allclose(kbp_coord, transformed)
    np.testing.assert_allclose(px_coord, c.to_pixels(transformed))

    c = PositionCalibration(PositionUnit.kbp, scale=0.42, origin=2.0)
    kbp_coord = np.array([-0.84, -0.336, 0.4788, 34.86])
    transformed = c.from_pixels(px_coord)
    np.testing.assert_allclose(kbp_coord, transformed)
    np.testing.assert_allclose(px_coord, c.to_pixels(transformed))

    c_flipped = PositionCalibration(PositionUnit.kbp, scale=-0.42, origin=2.0)
    transformed = c_flipped.from_pixels(px_coord)
    np.testing.assert_allclose(-kbp_coord, transformed)
    np.testing.assert_allclose(px_coord, c_flipped.to_pixels(transformed))

    assert c.scale == -c_flipped.scale
    assert c.pixelsize == c_flipped.pixelsize
