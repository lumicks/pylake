import pathlib

import numpy as np
import pytest

from lumicks.pylake.kymo import EmptyKymo


def test_kymo_slicing(test_kymo):
    kymo, ref = test_kymo
    ref_pixels = ref.metadata.pixels_per_line
    ref_lines = ref.metadata.lines_per_frame

    scan_time = (ref.timestamps.dt * ref.infowave.samples_per_pixel * ref_pixels) * 1e-9
    dead_time = (ref.timestamps.dt * ref.infowave.line_padding * 2) * 1e-9
    line_time = scan_time + dead_time

    # need to start slicing from the first sample _after_ the dead time
    deadtime_slice_offset = (1 + ref.infowave.line_padding) * ref.timestamps.dt * 1e-9

    assert kymo.get_image("red").shape == (ref_pixels, ref_lines)
    assert kymo.shape == (ref_pixels, ref_lines, 3)
    np.testing.assert_allclose(kymo.get_image("red").data, ref.image[:, :, 0])

    sliced = kymo[:]
    assert sliced.get_image("red").shape == (ref_pixels, ref_lines)
    np.testing.assert_allclose(sliced.get_image("red").data, ref.image[:, :, 0])

    # Anything will crop of the first frame
    # need to make sure you're past the deadtime
    # todo: for 2.0 define the start of the kymo as the start of the actual image, not deadtime
    sliced = kymo[f"{deadtime_slice_offset}s":]
    assert sliced.get_image("red").shape == (ref_pixels, ref_lines - 1)
    assert sliced.shape == (ref_pixels, ref_lines - 1, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, ref.image[:, 1:, 0])

    sliced = kymo["0s":]
    assert sliced.get_image("red").shape == (ref_pixels, ref_lines)
    np.testing.assert_allclose(sliced.get_image("red").data, ref.image[:, :, 0])

    num_lines = 2
    sliced = kymo["0s":f"{num_lines * line_time}s"]
    assert sliced.get_image("red").shape == (ref_pixels, num_lines)
    assert sliced.shape == (ref_pixels, num_lines, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, ref.image[:, :num_lines, 0])

    sliced = kymo["0s":f"-{line_time * 0.6}s"]
    assert sliced.get_image("red").shape == (ref_pixels, ref_lines - 1)
    np.testing.assert_allclose(sliced.get_image("red").data, ref.image[:, :-1, 0])

    sliced = kymo["0s":f"-{2 * line_time}s"]
    assert sliced.get_image("red").shape == (ref_pixels, ref_lines - 2)
    np.testing.assert_allclose(sliced.get_image("red").data, ref.image[:, : (ref_lines - 2), 0])

    # get a sliver of the next frame
    # stop needs to be > halfway the deadtime between lines
    sliced = kymo["0s":f"-{2 * line_time - deadtime_slice_offset}s"]
    assert sliced.get_image("red").shape == (ref_pixels, ref_lines - 1)
    np.testing.assert_allclose(sliced.get_image("red").data, ref.image[:, :-1, 0])

    # Two full frames
    sliced = kymo["0s":f"{2 * line_time}s"]
    assert sliced.get_image("red").shape == (ref_pixels, 2)
    np.testing.assert_allclose(sliced.get_image("red").data, ref.image[:, :2, 0])

    # Two full frames plus a bit
    sliced = kymo["0s":f"{2 * scan_time + 2 * dead_time + deadtime_slice_offset}s"]
    assert sliced.get_image("red").shape == (ref_pixels, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, ref.image[:, :3, 0])

    # slice from deadtime before first line until deadtime after first line
    sliced = kymo[f"{scan_time + dead_time / 2}s":f"{2 * line_time - dead_time / 2}s"]
    assert sliced.get_image("red").shape == (ref_pixels, 1)
    assert sliced.shape == (ref_pixels, 1, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, ref.image[:, 1:2, 0])

    # slice over entire kymo
    sliced = kymo["0s":f"{line_time * (ref.metadata.lines_per_frame + 1)}s"]
    assert sliced.get_image("red").shape == (ref_pixels, ref_lines)
    assert sliced.shape == (ref_pixels, ref_lines, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, ref.image[:, :, 0])

    with pytest.raises(IndexError, match="Scalar indexing is not supported, only slicing"):
        kymo["0s"]

    with pytest.raises(IndexError, match="Slice steps are not supported"):
        kymo["0s":"10s":"1s"]

    empty_kymograph = kymo["3s":"2s"]
    assert isinstance(empty_kymograph, EmptyKymo)

    empty_kymograph = kymo[f"{(ref.stop - ref.start) * 1e-9}s":]
    assert isinstance(empty_kymograph, EmptyKymo)

    with pytest.raises(RuntimeError, match="Can't get pixel timestamps if there are no pixels"):
        empty_kymograph.timestamps

    with pytest.raises(RuntimeError, match="Can't get pixel timestamps if there are no pixels"):
        empty_kymograph.export_tiff("test")

    with pytest.raises(RuntimeError, match="Cannot plot empty kymograph"):
        empty_kymograph.plot()

    assert empty_kymograph.get_image("red").shape == (ref_pixels, 0)
    assert empty_kymograph.infowave.data.size == 0
    assert empty_kymograph.shape == (ref_pixels, 0, 3)
    assert empty_kymograph.pixels_per_line == ref_pixels
    assert empty_kymograph.get_image("red").size == 0
    assert empty_kymograph.get_image("rgb").size == 0

    # Slicing by providing only the start time with a value greater than the start timestamp of the
    # very last line and less than or equal to the very last timestamp of the infowave created a
    # dysfunctional Kymo.
    assert isinstance(kymo[kymo.timestamps[2, -1] :], EmptyKymo)
    assert isinstance(kymo[f"{(kymo.timestamps[-1, -1] - kymo.start) * 1e-9}s":], EmptyKymo)


def test_downsampled_slice(test_kymo):
    """There was a regression bug that if a Kymo was downsampled and then sliced, it would undo the
    downsampling. For now, we just flag it as not implemented behaviour."""
    kymo, _ = test_kymo

    with pytest.raises(NotImplementedError):
        kymo.downsampled_by(time_factor=2)["1s":"2s"]


def test_kymo_crop(cropping_kymo):
    """Test basic cropping functionality"""
    kymo, ref = cropping_kymo
    px_size = ref.metadata.pixelsize_um[0]

    cropped = kymo.crop_by_distance(2 * px_size, 4 * px_size)
    ref_cropped = ref.image[2:4, :, 0]

    np.testing.assert_allclose(cropped.get_image("red"), ref_cropped)
    np.testing.assert_allclose(cropped.get_image("rgb")[:, :, 0], ref_cropped)
    np.testing.assert_allclose(cropped.get_image("rgb")[:, :, 1], np.zeros(ref_cropped.shape))
    np.testing.assert_allclose(cropped.get_image("green"), np.zeros(ref_cropped.shape))  # missing
    np.testing.assert_equal(cropped.timestamps, ref.timestamps.data[2:4, :])
    assert cropped.timestamps.dtype == np.int64
    np.testing.assert_allclose(cropped.pixelsize_um, kymo.pixelsize_um)
    np.testing.assert_allclose(cropped.line_time_seconds, kymo.line_time_seconds)
    np.testing.assert_allclose(cropped.pixels_per_line, 2)
    np.testing.assert_allclose(cropped._position_offset, 2 * px_size)

    with pytest.raises(ValueError, match="Cropping by negative positions not allowed"):
        kymo.crop_by_distance(-2 * px_size, px_size)

    with pytest.raises(ValueError, match="Cropping by negative positions not allowed"):
        kymo.crop_by_distance(px_size, -4 * px_size)

    with pytest.raises(IndexError, match="Cropped image would be empty"):
        kymo.crop_by_distance(5 * px_size, 2 * px_size)

    with pytest.raises(IndexError, match="Cropped image would be empty"):
        kymo.crop_by_distance(2 * px_size, 2 * px_size)

    with pytest.raises(IndexError, match="Cropped image would be empty"):
        kymo.crop_by_distance(20 * px_size, 21 * px_size)

    # Test rounding internally
    np.testing.assert_allclose(
        kymo.crop_by_distance(px_size * 1.6, px_size * 1.6).get_image("red"),
        ref.image[1:2, :, 0],
    )
    np.testing.assert_allclose(
        kymo.crop_by_distance(px_size * 1.6, px_size * 2.1).get_image("red"),
        ref.image[1:3, :, 0],
    )
    np.testing.assert_allclose(
        kymo.crop_by_distance(px_size * 2.1, px_size * 2.1).get_image("red"),
        ref.image[2:3, :, 0],
    )


def test_kymo_basepairs_crop(cropping_kymo):
    """Test basic cropping functionality"""
    kymo, ref = cropping_kymo
    kymo_bp = kymo.calibrate_to_kbp(1.000)
    px_size = kymo_bp.pixelsize[0]

    np.testing.assert_allclose(
        kymo_bp.crop_by_distance(px_size, 3 * px_size).get_image("red"), ref.image[1:3, :, 0]
    )
    np.testing.assert_allclose(
        kymo_bp.crop_by_distance(px_size, 3.5 * px_size).get_image("red"), ref.image[1:4, :, 0]
    )
    np.testing.assert_allclose(
        kymo_bp.crop_by_distance(px_size, 4 * px_size).get_image("red"), ref.image[1:4, :, 0]
    )


def test_kymo_crop_ds(cropping_kymo):
    """Test cropping interaction with downsampling"""

    kymo, ref = cropping_kymo
    px_size = ref.metadata.pixelsize_um[0]

    kymo_ds_pos = kymo.downsampled_by(position_factor=2)
    cropped = kymo_ds_pos.crop_by_distance(2 * px_size, 4 * px_size)
    np.testing.assert_allclose(cropped.get_image("red"), kymo_ds_pos.get_image("red")[1:2, :])
    np.testing.assert_allclose(cropped.timestamps, kymo_ds_pos.timestamps[1:2, :])
    np.testing.assert_allclose(cropped.pixelsize_um, kymo_ds_pos.pixelsize_um)
    np.testing.assert_allclose(cropped.line_time_seconds, kymo_ds_pos.line_time_seconds)
    np.testing.assert_allclose(cropped.pixels_per_line, 1)
    np.testing.assert_allclose(cropped._position_offset, 2 * px_size)

    kymo_ds_time = kymo.downsampled_by(time_factor=2)
    cropped = kymo_ds_time.crop_by_distance(2 * px_size, 4 * px_size)
    np.testing.assert_allclose(cropped.get_image("red"), kymo_ds_time.get_image("red")[2:4, :])
    np.testing.assert_allclose(cropped.pixelsize_um, kymo_ds_time.pixelsize_um)
    np.testing.assert_allclose(cropped.line_time_seconds, kymo_ds_time.line_time_seconds)
    np.testing.assert_allclose(cropped.pixels_per_line, 2)
    np.testing.assert_allclose(cropped._position_offset, 2 * px_size)

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
    check_order_of_operations(2, 1, 2 * px_size, 4 * px_size)
    check_order_of_operations(3, 1, 2 * px_size, 4 * px_size)
    check_order_of_operations(1, 2, 2 * px_size, 4 * px_size)
    check_order_of_operations(2, 2, 2 * px_size, 6 * px_size)
    check_order_of_operations(1, 3, 3 * px_size, 7 * px_size)


def test_kymo_slice_crop(cropping_kymo):
    """Test cropping after slicing"""
    kymo, ref = cropping_kymo
    ref_pixels = ref.metadata.pixels_per_line
    px_size = ref.metadata.pixelsize_um[0]

    scan_time = (ref.timestamps.dt * ref.infowave.samples_per_pixel * ref_pixels) * 1e-9
    dead_time = (ref.timestamps.dt * ref.infowave.line_padding * 2) * 1e-9
    line_time = scan_time + dead_time

    sliced_cropped = kymo[f"{line_time}s":f"{5 * line_time}s"].crop_by_distance(
        2 * px_size, 4 * px_size
    )
    np.testing.assert_equal(sliced_cropped.timestamps, ref.timestamps.data[2:4, 1:5])
    np.testing.assert_allclose(sliced_cropped.get_image("red"), ref.image[2:4, 1:5, 0])
    np.testing.assert_allclose(sliced_cropped._position_offset, 2 * px_size)

    np.testing.assert_equal(
        sliced_cropped._timestamps(reduce=np.min), kymo._timestamps(reduce=np.min)[2:4, 1:5]
    )


def test_incremental_offset(cropping_kymo):
    """Test whether cropping twice propagates the offset correctly"""
    kymo, ref = cropping_kymo
    px_size = ref.metadata.pixelsize_um[0]

    cropped = kymo.crop_by_distance(px_size, 4 * px_size)
    twice_cropped = cropped.crop_by_distance(px_size, 4 * px_size)

    np.testing.assert_allclose(
        twice_cropped.get_image("red"),
        ref.image[2:4, :, 0],
    )
    np.testing.assert_equal(
        twice_cropped.timestamps,
        ref.timestamps.data[2:4, :],
    )
    np.testing.assert_allclose(twice_cropped.pixelsize_um, kymo.pixelsize_um)
    np.testing.assert_allclose(twice_cropped.line_time_seconds, kymo.line_time_seconds)
    np.testing.assert_allclose(twice_cropped.pixels_per_line, 2)
    np.testing.assert_allclose(twice_cropped._position_offset, 2 * px_size)


@pytest.mark.parametrize(
    "color, ref_locations, algorithm, kwargs",
    [
        ("red", (7.65, 17.7), "brightness", {"threshold_percentile": 70}),
        ("green", (7.65, 17.65), "brightness", {"threshold_percentile": 70}),
        ("blue", (7.25, 20.9), "brightness", {"threshold_percentile": 70}),
        ("blue", (7.25, 20.45), "brightness", {"threshold_percentile": 20}),
        ("red", (7.4, 17.7), "template", {}),
        ("green", (7.35, 17.75), "template", {}),
        ("blue", (7.05, 18.45), "template", {}),
        ("blue", (7.2, 18.3), "template", {"threshold_percentile": 20}),
        ("blue", (6.9, 18.45), "template", {"allow_movement": True}),
        ("red", (7.65 + 1.5, 17.7 - 1.5), "brightness", {"extra_cropping": 1.5}),
    ],
)
def test_bead_crop(bead_kymo, color, ref_locations, algorithm, kwargs):
    np.testing.assert_allclose(
        bead_kymo.estimate_bead_edges(4.89, algorithm=algorithm, channel=color, **kwargs),
        ref_locations,
    )
    np.testing.assert_allclose(
        bead_kymo.crop_beads(4.89, algorithm=algorithm, channel=color, **kwargs).size_um,
        np.diff(ref_locations),
        atol=bead_kymo.pixelsize_um[0],
    )


def test_bead_crop_invalid_algorithm(bead_kymo):
    with pytest.raises(ValueError, match="Unrecognized algorithm godot"):
        bead_kymo.crop_beads(4.89, algorithm="godot", channel="green")


def test_bead_crop_invalid_extra_crop(bead_kymo):
    extra_cropping = (17.65 - 7.65) / 2 - bead_kymo.pixelsize_um[0]
    bead_kymo.crop_beads(
        4.89, algorithm="brightness", channel="green", extra_cropping=extra_cropping
    )

    with pytest.raises(
        RuntimeError,
        match=r"Detected bead edges in combination with chosen extra cropping \(5\.00\)",
    ):
        extra_cropping = (17.65 - 7.65) / 2
        bead_kymo.crop_beads(
            4.89, algorithm="brightness", channel="green", extra_cropping=extra_cropping
        )
