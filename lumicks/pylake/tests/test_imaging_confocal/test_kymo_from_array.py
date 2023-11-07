import itertools

import numpy as np
import pytest
import matplotlib.pyplot as plt

from lumicks.pylake.kymo import _kymo_from_array

timestamp_err_msg = (
    "Per-pixel timestamps are not implemented. "
    "Line timestamps are still available, however. See: `Kymo.line_time_seconds`."
)

colors = ("red", "green", "blue")


def make_kymo_from_array(kymo, image, color_format, no_pxsize=False):
    start = kymo._timestamps(np.min)[0, 0]
    exposure_time_sec = np.diff(kymo.line_timestamp_ranges(include_dead_time=False)[0])[0] * 1e-9

    return _kymo_from_array(
        image,
        color_format,
        kymo.line_time_seconds,
        start=start,
        exposure_time_seconds=exposure_time_sec,
        name="reconstructed",
        pixel_size_um=None if no_pxsize else kymo.pixelsize_um[0],
    )


@pytest.mark.parametrize("crop", [False, True])
def test_from_array(test_kymo, crop):
    kymo, ref = test_kymo
    arr_kymo = make_kymo_from_array(kymo, kymo.get_image("rgb"), "rgb")

    if crop:
        arr_kymo = arr_kymo.crop_by_distance(0.0, 4 * ref.metadata.pixelsize_um[0])
    ref_image = ref.image[:4, :, :] if crop else ref.image.data
    ref_shape = (4,) if crop else kymo._reconstruction_shape

    np.testing.assert_equal(arr_kymo.get_image("rgb"), ref_image)
    np.testing.assert_equal(arr_kymo._reconstruction_shape, ref_shape)

    with pytest.raises(NotImplementedError, match=timestamp_err_msg):
        arr_kymo.pixel_time_seconds

    np.testing.assert_equal(kymo.line_time_seconds, arr_kymo.line_time_seconds)
    np.testing.assert_equal(
        kymo.line_timestamp_ranges(include_dead_time=False),
        arr_kymo.line_timestamp_ranges(include_dead_time=False),
    )
    np.testing.assert_equal(
        kymo.line_timestamp_ranges(include_dead_time=True),
        arr_kymo.line_timestamp_ranges(include_dead_time=True),
    )

    np.testing.assert_equal(kymo.pixelsize_um, arr_kymo.pixelsize_um)
    np.testing.assert_equal(kymo.pixelsize, arr_kymo.pixelsize)

    assert arr_kymo._metadata.center_point_um == {key: None for key in ("x", "y", "z")}
    assert arr_kymo._metadata.num_frames == 0

    with pytest.raises(
        NotImplementedError,
        match="Slicing is not implemented for kymographs derived from image stacks.",
    ):
        arr_kymo["0s":"0.5s"]

    with pytest.raises(
        NotImplementedError,
        match="Slicing is not implemented for kymographs derived from image stacks.",
    ):
        arr_kymo.crop_and_calibrate("red")


@pytest.mark.parametrize("position_factor, time_factor", [[1, 1], [1, 2], [2, 1], [2, 2]])
def test_downsampling(test_kymo, position_factor, time_factor):
    kymo, _ = test_kymo

    arr_kymo_ds = make_kymo_from_array(kymo, kymo.get_image("rgb"), "rgb").downsampled_by(
        position_factor=position_factor, time_factor=time_factor
    )
    kymo_ds = kymo.downsampled_by(position_factor=position_factor, time_factor=time_factor)

    if time_factor > 1:
        with pytest.raises(NotImplementedError, match="no longer available after downsampling"):
            arr_kymo_ds.line_timestamp_ranges()
    else:
        # Note, kymo line timestamp ranges should *not* modify under downsampling
        np.testing.assert_allclose(
            arr_kymo_ds.line_timestamp_ranges(), kymo.line_timestamp_ranges()
        )

    np.testing.assert_allclose(arr_kymo_ds.pixelsize_um, kymo_ds.pixelsize_um)
    np.testing.assert_allclose(arr_kymo_ds.line_time_seconds, kymo_ds.line_time_seconds)


def test_save_tiff(tmpdir_factory, test_kymo):
    from os import stat

    tmpdir = tmpdir_factory.mktemp("pylake")

    kymo, _ = test_kymo

    for no_pxsize in (True, False):
        arr_kymo = make_kymo_from_array(kymo, kymo.get_image("rgb"), "rgb", no_pxsize=no_pxsize)

        with pytest.warns(UserWarning):
            arr_kymo.export_tiff(f"{tmpdir}/kymo1.tiff")
            assert stat(f"{tmpdir}/kymo1.tiff").st_size > 0


def test_from_array_no_pixelsize(test_kymo):
    kymo, _ = test_kymo
    arr_kymo = make_kymo_from_array(kymo, kymo.get_image("rgb"), "rgb", no_pxsize=True)

    np.testing.assert_equal(kymo.get_image("rgb"), arr_kymo.get_image("rgb"))
    np.testing.assert_equal(kymo._reconstruction_shape, arr_kymo._reconstruction_shape)

    with pytest.raises(NotImplementedError, match=timestamp_err_msg):
        arr_kymo.pixel_time_seconds

    np.testing.assert_equal(kymo.line_time_seconds, arr_kymo.line_time_seconds)
    np.testing.assert_equal(
        kymo.line_timestamp_ranges(include_dead_time=False),
        arr_kymo.line_timestamp_ranges(include_dead_time=False),
    )
    np.testing.assert_equal(
        kymo.line_timestamp_ranges(include_dead_time=True),
        arr_kymo.line_timestamp_ranges(include_dead_time=True),
    )

    assert arr_kymo.pixelsize_um == [None]
    assert arr_kymo.pixelsize == [1.0]
    assert arr_kymo._calibration.unit == "pixel"

    assert arr_kymo._metadata.center_point_um == {key: None for key in ("x", "y", "z")}
    assert arr_kymo._metadata.num_frames == 0
    assert arr_kymo._motion_blur_constant is None


def test_throw_on_file_access(test_kymo):
    kymo, _ = test_kymo
    arr_kymo = make_kymo_from_array(kymo, kymo.get_image("rgb"), "rgb")

    attributes = (
        [f"{color}_power" for color in colors]
        + [f"{color}_photon_count" for color in colors]
        + ["infowave", "sted_power"]
    )
    for attribute in attributes:
        with pytest.raises(ValueError, match="There is no .h5 file associated with this Kymo"):
            getattr(arr_kymo, attribute)

    with pytest.raises(ValueError, match="There is no force data associated with this Kymo"):
        arr_kymo.plot_with_force("1x", "red")


def test_from_array_fewer_channels(test_kymo):
    kymo, _ = test_kymo
    rgb_image = kymo.get_image("rgb")

    # single-channel data
    for j, color_channel in enumerate(colors):
        arr_kymo = make_kymo_from_array(kymo, rgb_image[:, :, j], color_channel[0], no_pxsize=True)

        for channel in colors:
            if channel == color_channel:
                np.testing.assert_equal(arr_kymo.get_image(channel), kymo.get_image(channel))
            else:
                np.testing.assert_equal(arr_kymo.get_image(channel), 0)

    # two-channel data
    for channels in itertools.permutations(colors, 2):
        color_format = "".join([c[0] for c in channels])
        sub_image = np.stack([rgb_image[:, :, "rgb".index(c)] for c in color_format], axis=2)
        arr_kymo = make_kymo_from_array(kymo, sub_image, color_format, no_pxsize=True)

        for channel in colors:
            if channel in channels:
                np.testing.assert_equal(arr_kymo.get_image(channel), kymo.get_image(channel))
            else:
                np.testing.assert_equal(arr_kymo.get_image(channel), 0)


def test_color_format(test_kymo):
    kymo, _ = test_kymo
    image = kymo.get_image("rgb")
    with pytest.raises(
        ValueError, match="Invalid color format 'rgp'. Only 'r', 'g', and 'b' are valid components."
    ):
        arr_kymo = make_kymo_from_array(kymo, image, "rgp")

    with pytest.raises(
        ValueError, match="Color format 'r' specifies 1 channel for a 3 channel image."
    ):
        arr_kymo = make_kymo_from_array(kymo, image, "r")

    with pytest.raises(
        ValueError, match="Color format 'rb' specifies 2 channels for a 3 channel image."
    ):
        arr_kymo = make_kymo_from_array(kymo, image, "rb")

    with pytest.raises(
        ValueError, match="Color format 'rgb' specifies 3 channels for a 2 channel image."
    ):
        arr_kymo = make_kymo_from_array(kymo, image[:, :, :2], "rgb")
