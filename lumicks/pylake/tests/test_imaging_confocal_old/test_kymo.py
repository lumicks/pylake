import re

import numpy as np
import pytest
import matplotlib.pyplot as plt

from lumicks import pylake
from lumicks.pylake.kymo import EmptyKymo, _default_line_time_factory
from lumicks.pylake.channel import Slice, Continuous, TimeSeries, empty_slice
from lumicks.pylake.adjustments import ColorAdjustment

from ..data.mock_confocal import generate_kymo


def with_offset(t, start_time=1592916040906356300):
    return np.array(t, dtype=np.int64) + start_time


def test_kymo_slicing(test_kymos):
    kymo = test_kymos["Kymo1"]
    kymo_reference = np.transpose(
        [[2, 0, 0, 0, 2], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]]
    )

    assert kymo.get_image("red").shape == (5, 4)
    assert kymo.shape == (5, 4, 3)
    np.testing.assert_allclose(kymo.get_image("red").data, kymo_reference)

    sliced = kymo[:]
    assert sliced.get_image("red").shape == (5, 4)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference)

    scan_time, dead_time = 0.9375, 0.625

    sliced = kymo["0.1s":]  # Anything will crop of the first frame
    assert sliced.get_image("red").shape == (5, 3)
    assert sliced.shape == (5, 3, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, 1:])

    sliced = kymo["0s":]
    assert sliced.get_image("red").shape == (5, 4)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference)

    sliced = kymo["0s":f"{2 * scan_time + dead_time}s"]
    assert sliced.get_image("red").shape == (5, 2)
    assert sliced.shape == (5, 2, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, :2])

    sliced = kymo["0s":f"-{scan_time}s"]
    assert sliced.get_image("red").shape == (5, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, :-1])

    sliced = kymo["0s":f"-{2 * scan_time + 2 * dead_time}s"]
    assert sliced.get_image("red").shape == (5, 2)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, :-2])

    sliced = kymo["0s":f"-{2 * scan_time + dead_time - 0.1}s"]  # Get a sliver of next frame
    assert sliced.get_image("red").shape == (5, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, :-1])

    sliced = kymo["0s":f"{2 * scan_time + 2 * dead_time}s"]  # Two full frames
    assert sliced.get_image("red").shape == (5, 2)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, :2])

    sliced = kymo["0s":f"{2 * scan_time + 2 * dead_time + 0.01}s"]  # Two full frames plus a bit
    assert sliced.get_image("red").shape == (5, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, :3])

    sliced = kymo["1s":"2s"]
    assert sliced.get_image("red").shape == (5, 1)
    assert sliced.shape == (5, 1, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, 1:2])

    sliced = kymo["0s":"10s"]
    assert sliced.get_image("red").shape == (5, 4)
    assert sliced.shape == (5, 4, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, 0:10])

    with pytest.raises(IndexError):
        kymo["0s"]

    with pytest.raises(IndexError):
        kymo["0s":"10s":"1s"]

    empty_kymograph = kymo["3s":"2s"]
    assert isinstance(empty_kymograph, EmptyKymo)

    empty_kymograph = kymo["5s":]
    assert isinstance(empty_kymograph, EmptyKymo)

    with pytest.raises(RuntimeError):
        empty_kymograph.timestamps

    with pytest.raises(RuntimeError):
        empty_kymograph.export_tiff("test")

    with pytest.raises(RuntimeError):
        empty_kymograph.plot()

    assert empty_kymograph.get_image("red").shape == (5, 0)
    assert empty_kymograph.infowave.data.size == 0
    assert empty_kymograph.shape == (5, 0, 3)
    assert empty_kymograph.pixels_per_line == 5
    assert empty_kymograph.get_image("red").size == 0
    assert empty_kymograph.get_image("rgb").size == 0

    kymo = test_kymos["slicing_regression"]
    assert isinstance(kymo["23.0s":], EmptyKymo)
    assert isinstance(kymo["24.2s":], EmptyKymo)


def test_export_tiff(tmp_path, test_kymos, grab_tiff_tags):
    from os import stat

    kymo = test_kymos["Kymo1"]
    kymo.export_tiff(tmp_path / "kymo1.tiff")
    assert stat(tmp_path / "kymo1.tiff").st_size > 0

    # Check if tags were properly stored, i.e. test functionality of `_tiff_image_metadata()`,
    # `_tiff_timestamp_ranges()` and `_tiff_writer_kwargs()`
    tiff_tags = grab_tiff_tags(tmp_path / "kymo1.tiff")
    assert len(tiff_tags) == 1
    for tags, timestamp_range in zip(tiff_tags, kymo._tiff_timestamp_ranges()):
        assert tags["ImageDescription"] == kymo._tiff_image_metadata()
        assert tags["DateTime"] == f"{timestamp_range[0]}:{timestamp_range[1]}"
        assert tags["Software"] == kymo._tiff_writer_kwargs()["software"]
        np.testing.assert_allclose(
            tags["XResolution"][0] / tags["XResolution"][1],
            kymo._tiff_writer_kwargs()["resolution"][0],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            tags["YResolution"][0] / tags["YResolution"][1],
            kymo._tiff_writer_kwargs()["resolution"][1],
            rtol=1e-1,
        )
        assert tags["ResolutionUnit"] == 3  # 3 = Centimeter


def test_downsampled_slice():
    """There was a regression bug that if a Kymo was downsampled and then sliced, it would undo the
    downsampling. For now, we just flag it as not implemented behaviour."""
    kymo = generate_kymo(
        "Mock",
        image=np.array([[2, 2], [2, 2]], dtype=np.uint8),
        pixel_size_nm=1,
        start=100,
        dt=7,
        samples_per_pixel=5,
        line_padding=2,
    )

    with pytest.raises(NotImplementedError):
        kymo.downsampled_by(time_factor=2)["1s":"2s"]


def test_kymo_crop():
    """Test basic cropping functionality"""
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8,
    )

    # Test basic functionality
    pixel_size_nm = 2
    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=pixel_size_nm,
        start=with_offset(100),
        dt=5,
        samples_per_pixel=5,
        line_padding=2,
    )
    cropped = kymo.crop_by_distance(4e-3, 8e-3)
    ref_img = np.array(
        [[12.0, 0.0, 0.0, 0.0, 12.0, 6.0, 0.0], [0.0, 12.0, 12.0, 12.0, 0.0, 6.0, 0.0]]
    )
    np.testing.assert_allclose(cropped.get_image("red"), ref_img)
    np.testing.assert_allclose(cropped.get_image("rgb")[:, :, 0], ref_img)
    np.testing.assert_allclose(cropped.get_image("rgb")[:, :, 1], np.zeros(ref_img.shape))
    np.testing.assert_allclose(cropped.get_image("green"), np.zeros(ref_img.shape))  # missing
    np.testing.assert_equal(
        cropped.timestamps,
        with_offset([[170, 315, 460, 605, 750, 895, 1040], [195, 340, 485, 630, 775, 920, 1065]]),
    )
    assert cropped.timestamps.dtype == np.int64
    np.testing.assert_allclose(cropped.pixelsize_um, kymo.pixelsize_um)
    np.testing.assert_allclose(cropped.line_time_seconds, kymo.line_time_seconds)
    np.testing.assert_allclose(cropped.pixels_per_line, 2)
    np.testing.assert_allclose(cropped._position_offset, 4e-3)

    with pytest.raises(ValueError, match="Cropping by negative positions not allowed"):
        kymo.crop_by_distance(-4e3, 1e3)

    with pytest.raises(ValueError, match="Cropping by negative positions not allowed"):
        kymo.crop_by_distance(1e3, -4e3)

    with pytest.raises(IndexError, match="Cropped image would be empty"):
        kymo.crop_by_distance(5e-3, 2e-3)

    with pytest.raises(IndexError, match="Cropped image would be empty"):
        kymo.crop_by_distance(2e-3, 2e-3)

    with pytest.raises(IndexError, match="Cropped image would be empty"):
        kymo.crop_by_distance(20e3, 21e3)

    # Test rounding internally
    # fmt: off
    np.testing.assert_allclose(
        kymo.crop_by_distance(pixel_size_nm * 1.6 * 1e-3, pixel_size_nm * 1.6 * 1e-3).get_image("red"),
        image[1:2, :],
    )
    np.testing.assert_allclose(
        kymo.crop_by_distance(pixel_size_nm * 1.6 * 1e-3, pixel_size_nm * 2.1 * 1e-3).get_image("red"),
        image[1:3, :],
    )
    np.testing.assert_allclose(
        kymo.crop_by_distance(pixel_size_nm * 2.1 * 1e-3, pixel_size_nm * 2.1 * 1e-3).get_image("red"),
        image[2:3, :],
    )
    # fmt: on

    # Test cropping in base pairs
    kymo_bp = kymo.calibrate_to_kbp(1.000)  # pixelsize = 0.2 kbp
    np.testing.assert_allclose(
        kymo_bp.crop_by_distance(0.2, 0.6).get_image("red"),
        [[0, 0, 0, 0, 0, 6, 0], [12, 0, 0, 0, 12, 6, 0]],
    )
    np.testing.assert_allclose(
        kymo_bp.crop_by_distance(0.2, 0.7).get_image("red"),
        [[0, 0, 0, 0, 0, 6, 0], [12, 0, 0, 0, 12, 6, 0], [0, 12, 12, 12, 0, 6, 0]],
    )
    np.testing.assert_allclose(
        kymo_bp.crop_by_distance(0.2, 0.8).get_image("red"),
        [[0, 0, 0, 0, 0, 6, 0], [12, 0, 0, 0, 12, 6, 0], [0, 12, 12, 12, 0, 6, 0]],
    )


def test_kymo_crop_ds():
    """Test cropping interaction with downsampling"""

    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [12, 12, 12, 12, 0, 6, 0],
            [24, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8,
    )

    pixel_size_nm = 2
    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=pixel_size_nm,
        start=with_offset(100),
        dt=5,
        samples_per_pixel=5,
        line_padding=2,
    )

    kymo_ds_pos = kymo.downsampled_by(position_factor=2)
    cropped = kymo_ds_pos.crop_by_distance(4e-3, 8e-3)
    np.testing.assert_allclose(cropped.get_image("red"), kymo_ds_pos.get_image("red")[1:2, :])
    np.testing.assert_allclose(cropped.timestamps, kymo_ds_pos.timestamps[1:2, :])
    np.testing.assert_allclose(cropped.pixelsize_um, kymo_ds_pos.pixelsize_um)
    np.testing.assert_allclose(cropped.line_time_seconds, kymo_ds_pos.line_time_seconds)
    np.testing.assert_allclose(cropped.pixels_per_line, 1)
    np.testing.assert_allclose(cropped._position_offset, 4e-3)

    kymo_ds_time = kymo.downsampled_by(time_factor=2)
    cropped = kymo_ds_time.crop_by_distance(4e-3, 8e-3)
    np.testing.assert_allclose(cropped.get_image("red"), kymo_ds_time.get_image("red")[2:4, :])
    np.testing.assert_allclose(cropped.pixelsize_um, kymo_ds_time.pixelsize_um)
    np.testing.assert_allclose(cropped.line_time_seconds, kymo_ds_time.line_time_seconds)
    np.testing.assert_allclose(cropped.pixels_per_line, 2)
    np.testing.assert_allclose(cropped._position_offset, 4e-3)

    def check_order_of_operations(time_factor, pos_factor, crop_x, crop_y):
        crop_ds = kymo.crop_by_distance(crop_x, crop_y).downsampled_by(time_factor, pos_factor)
        ds_crop = kymo.downsampled_by(time_factor, pos_factor).crop_by_distance(crop_x, crop_y)

        np.testing.assert_allclose(crop_ds.get_image("red"), ds_crop.get_image("red"))
        np.testing.assert_allclose(crop_ds.line_time_seconds, ds_crop.line_time_seconds)
        np.testing.assert_allclose(crop_ds.pixelsize_um, ds_crop.pixelsize_um)
        np.testing.assert_allclose(crop_ds.pixels_per_line, ds_crop.pixels_per_line)
        np.testing.assert_allclose(crop_ds._position_offset, ds_crop._position_offset)

        if time_factor == 1:
            np.testing.assert_allclose(crop_ds.get_image("red"), ds_crop.get_image("red"))

    # Note that the order of operations check only makes sense for where the cropping happens on
    # a multiple of the downsampling.
    check_order_of_operations(2, 1, 4e-3, 8e-3)
    check_order_of_operations(3, 1, 4e-3, 8e-3)
    check_order_of_operations(1, 2, 4e-3, 8e-3)
    check_order_of_operations(2, 2, 4e-3, 12e-3)
    check_order_of_operations(1, 3, 6e-3, 14e-3)


def test_kymo_slice_crop():
    """Test cropping after slicing"""
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8,
    )
    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=4,
        start=int(100e9),
        dt=int(5e9),
        samples_per_pixel=5,
        line_padding=2,
    )

    sliced_cropped = kymo["245s":"725s"].crop_by_distance(8e-3, 14e-3)
    np.testing.assert_equal(
        sliced_cropped.timestamps, [[460e9, 605e9, 750e9], [485e9, 630e9, 775e9]]
    )
    np.testing.assert_allclose(sliced_cropped.get_image("red"), [[0, 0, 12], [12, 12, 0]])
    np.testing.assert_allclose(sliced_cropped._position_offset, 8e-3)
    np.testing.assert_equal(
        sliced_cropped._timestamps(reduce=np.min), [[450e9, 595e9, 740e9], [475e9, 620e9, 765e9]]
    )


def test_incremental_offset():
    """Test whether cropping twice propagates the offset correctly"""
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8,
    )

    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=2,
        start=with_offset(100),
        dt=5,
        samples_per_pixel=5,
        line_padding=2,
    )
    cropped = kymo.crop_by_distance(2e-3, 8e-3)
    twice_cropped = cropped.crop_by_distance(2e-3, 8e-3)
    np.testing.assert_allclose(
        twice_cropped.get_image("red"),
        [[12.0, 0.0, 0.0, 0.0, 12.0, 6.0, 0.0], [0.0, 12.0, 12.0, 12.0, 0.0, 6.0, 0.0]],
    )
    np.testing.assert_equal(
        twice_cropped.timestamps,
        with_offset([[170, 315, 460, 605, 750, 895, 1040], [195, 340, 485, 630, 775, 920, 1065]]),
    )
    np.testing.assert_allclose(twice_cropped.pixelsize_um, kymo.pixelsize_um)
    np.testing.assert_allclose(twice_cropped.line_time_seconds, kymo.line_time_seconds)
    np.testing.assert_allclose(twice_cropped.pixels_per_line, 2)
    np.testing.assert_allclose(twice_cropped._position_offset, 4e-3)


def test_slice_timestamps():
    """Test slicing with realistically sized timestamps (this tests against floating point errors
    induced in kymograph reconstruction)"""
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
        ],
        dtype=np.uint8,
    )

    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=4,
        start=1623965975045144000,
        dt=int(1e9),
        samples_per_pixel=5,
        line_padding=2,
    )

    # Kymo line is 3 * 5 samples long, while there is 2 pixel padding on each side.
    # Starting pixel time stamps relative to the original are as follows:
    # [[2  21  40  59  78  97 116]
    #  [7  26  45  64  83 102 121]
    # [12  31  50  69  88 107 126]]
    ref_ts = kymo.timestamps
    sliced = kymo["2s":"120s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts)
    sliced = kymo["3s":"120s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts[:, 1:])
    sliced = kymo["21s":"120s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts[:, 1:])
    sliced = kymo["22s":"120s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts[:, 2:])
    sliced = kymo["22s":"97s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts[:, 2:5])
    sliced = kymo["22s":"98s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts[:, 2:6])
    sliced = kymo["0s":"98s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts[:, :6])


def test_calibrate_to_kbp():
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 6, 6, 0, 0, 12],
            [6, 12, 12, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=100,
        start=1623965975045144000,
        dt=int(1e9),
        samples_per_pixel=5,
        line_padding=2,
    )

    kymo_bp = kymo.calibrate_to_kbp(12.000)

    # test that default calibration is in microns
    assert kymo._calibration.unit == "um"
    assert kymo._calibration.value == 0.1

    # test that calibration is stored as kilobase-pairs
    assert kymo_bp._calibration.unit == "kbp"
    np.testing.assert_allclose(kymo_bp._calibration.value, 2.0)

    # test conversion from microns to calibration units
    np.testing.assert_allclose(kymo._calibration.value * kymo._num_pixels[0], 0.6)
    np.testing.assert_allclose(kymo.pixelsize, 0.1)
    np.testing.assert_allclose(kymo_bp._calibration.value * kymo._num_pixels[0], 12.0)
    np.testing.assert_allclose(kymo_bp.pixelsize, 2.0)

    # test that all factories were forwarded from original instance
    def check_factory_forwarding(kymo1, kymo2, check_timestamps):
        assert kymo1._image_factory == kymo2._image_factory
        assert kymo1._timestamp_factory == kymo2._timestamp_factory
        assert kymo1._line_time_factory == kymo2._line_time_factory
        assert kymo1._pixelsize_factory == kymo2._pixelsize_factory
        assert kymo1._pixelcount_factory == kymo2._pixelcount_factory
        np.testing.assert_allclose(kymo1.get_image("red"), kymo2.get_image("red"))
        if check_timestamps:
            np.testing.assert_allclose(kymo1.timestamps, kymo2.timestamps)

    # check that calibration is supported for any processing (downsampling/cropping)
    # and that data remains the same after calibration
    ds_kymo_time = kymo.downsampled_by(time_factor=2)
    ds_kymo_pos = kymo.downsampled_by(position_factor=3)
    ds_kymo_both = kymo.downsampled_by(time_factor=2, position_factor=3)
    sliced_kymo = kymo["0s":"110s"]
    cropped_kymo = kymo.crop_by_distance(0, 0.5)

    ds_kymo_time_bp = ds_kymo_time.calibrate_to_kbp(12.000)
    ds_kymo_pos_bp = ds_kymo_pos.calibrate_to_kbp(12.000)  # total length does not change
    ds_kymo_both_bp = ds_kymo_both.calibrate_to_kbp(12.000)
    sliced_kymo_bp = sliced_kymo.calibrate_to_kbp(12.000)
    cropped_kymo_bp = cropped_kymo.calibrate_to_kbp(int(12.000 * (5 / 6)))

    check_factory_forwarding(kymo, kymo_bp, True)
    check_factory_forwarding(ds_kymo_time, ds_kymo_time_bp, False)
    check_factory_forwarding(ds_kymo_pos, ds_kymo_pos_bp, True)
    check_factory_forwarding(ds_kymo_both, ds_kymo_both_bp, False)
    check_factory_forwarding(sliced_kymo, sliced_kymo_bp, True)
    check_factory_forwarding(cropped_kymo, cropped_kymo_bp, True)

    # if properly calibrated, cropping should not change pixel size
    np.testing.assert_allclose(kymo_bp.pixelsize[0], cropped_kymo_bp.pixelsize[0])
    # but will change total length
    np.testing.assert_allclose(
        kymo_bp._calibration.value * kymo._num_pixels[0] * 5 / 6,
        cropped_kymo_bp._calibration.value * cropped_kymo_bp._num_pixels[0],
    )


def test_kymo_plot_rgb_absolute_color_adjustment(test_kymos):
    """Tests whether we can set an absolute color range for the RGB plot."""
    kymo = test_kymos["Kymo1"]

    fig = plt.figure()
    lb, ub = np.array([1, 2, 3]), np.array([2, 3, 4])
    kymo.plot(channel="rgb", adjustment=ColorAdjustment(lb, ub, mode="absolute"))
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(
        image.get_array(), np.clip((kymo.get_image("rgb") - lb) / (ub - lb), 0, 1)
    )
    plt.close(fig)


def test_kymo_plot_rgb_percentile_color_adjustment(test_kymos):
    """Tests whether we can set a percentile color range for the RGB plot."""
    kymo = test_kymos["Kymo1"]

    fig = plt.figure()
    lb, ub = np.array([10, 10, 10]), np.array([80, 80, 80])
    kymo.plot(channel="rgb", adjustment=ColorAdjustment(lb, ub, mode="percentile"))
    image = plt.gca().get_images()[0]
    bounds = np.array(
        [
            np.percentile(img, [mini, maxi])
            for img, mini, maxi in zip(np.moveaxis(kymo.get_image("rgb"), 2, 0), lb, ub)
        ]
    )
    lb, ub = (b for b in np.moveaxis(bounds, 1, 0))
    np.testing.assert_allclose(
        image.get_array(), np.clip((kymo.get_image("rgb") - lb) / (ub - lb), 0, 1)
    )
    plt.close(fig)


def test_kymo_plot_single_channel_absolute_color_adjustment(test_kymos):
    """Tests whether we can set an absolute color range for a single channel plot."""
    kymo = test_kymos["Kymo1"]

    lbs, ubs = np.array([1, 2, 3]), np.array([2, 3, 4])
    for lb, ub, channel in zip(lbs, ubs, ("red", "green", "blue")):
        # Test whether setting RGB values and then sampling one of them works correctly.
        fig = plt.figure()
        kymo.plot(channel=channel, adjustment=ColorAdjustment(lbs, ubs, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), kymo.get_image(channel))
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)

        # Test whether setting a single color works correctly (should use the same for R G and B).
        fig = plt.figure()
        kymo.plot(channel=channel, adjustment=ColorAdjustment(lb, ub, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), kymo.get_image(channel))
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)


@pytest.mark.parametrize("crop", [False, True])
def test_flip_kymo(test_kymos, crop):
    kymo = test_kymos["Kymo1"]
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
