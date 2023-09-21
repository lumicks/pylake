import re

import numpy as np
import pytest

from lumicks.pylake.detail.confocal import timestamp_mean


def test_downsampled_kymo_time(downsampling_kymo, downsampled_results):
    """Test downsampling over the temporal axis"""

    kymo, ref = downsampling_kymo
    factor, _, ds_image, *_ = downsampled_results
    ref_line_time = (
        (padding := ref.infowave.line_padding)
        * ref.timestamps.dt
        * (ref.infowave.samples_per_pixel * ref.metadata.pixels_per_line + padding + padding)
        / 1e9
    )

    kymo_ds = kymo.downsampled_by(time_factor=factor)

    assert kymo_ds.name == "downsampler"
    np.testing.assert_allclose(kymo_ds.get_image("red"), ds_image)
    np.testing.assert_allclose(kymo_ds.shape, kymo_ds.get_image("rgb").shape)
    np.testing.assert_allclose(kymo_ds.start, kymo.start)
    np.testing.assert_allclose(kymo_ds.pixelsize_um, ref.metadata.pixelsize_um)
    np.testing.assert_allclose(kymo_ds.pixelsize, ref.metadata.pixelsize_um[0])
    np.testing.assert_allclose(kymo_ds.line_time_seconds, ref_line_time)
    assert not kymo_ds.contiguous

    with pytest.raises(
        NotImplementedError,
        match=re.escape("Per-pixel timestamps are no longer available after downsampling"),
    ):
        kymo_ds.timestamps

    with pytest.raises(
        NotImplementedError,
        match=re.escape("No motion blur constant was defined for this kymograph"),
    ):
        kymo_ds.motion_blur_constant

    with pytest.raises(
        NotImplementedError,
        match="Per-pixel timestamps are no longer available after downsampling",
    ):
        kymo_ds.pixel_time_seconds

    # Verify that we can pass a different reduce function
    np.testing.assert_allclose(
        kymo.downsampled_by(time_factor=factor, reduce=np.mean).get_image("red"),
        ds_image / 2,
    )


def test_downsampled_kymo_position(downsampling_kymo, downsampled_results):
    """Test downsampling over the spatial axis"""

    kymo, ref = downsampling_kymo
    _, factor, _, ds_image, _ = downsampled_results
    n = int(np.floor(ref.metadata.pixels_per_line / factor))
    downsampled_timestamps = np.vstack(
        [
            timestamp_mean(kymo.timestamps[j * factor : (j + 1) * factor, :], axis=0)
            for j in range(n)
        ]
    )

    kymo_ds = kymo.downsampled_by(position_factor=factor)

    assert kymo_ds.name == "downsampler"
    np.testing.assert_allclose(kymo_ds.get_image("red"), ds_image)
    np.testing.assert_equal(kymo_ds.timestamps, downsampled_timestamps)
    np.testing.assert_allclose(kymo_ds.start, kymo.start)
    np.testing.assert_allclose(
        kymo_ds.pixelsize_um,
        np.array(ref.metadata.pixelsize_um) * factor,
    )
    np.testing.assert_allclose(kymo_ds.pixelsize, ref.metadata.pixelsize_um[0] * factor)
    np.testing.assert_allclose(kymo_ds.line_time_seconds, kymo.line_time_seconds)
    assert kymo_ds.contiguous

    # We lost one line while downsampling
    np.testing.assert_allclose(kymo_ds.size_um[0], kymo.size_um[0] - kymo.pixelsize_um[0])

    np.testing.assert_allclose(kymo_ds.pixel_time_seconds, kymo.pixel_time_seconds * 2)


def test_downsampled_kymo_both_axes(downsampling_kymo, downsampled_results):
    kymo, ref = downsampling_kymo
    t_factor, p_factor, *_, ds_image = downsampled_results
    ref_line_time = (
        (padding := ref.infowave.line_padding)
        * ref.timestamps.dt
        * (ref.infowave.samples_per_pixel * ref.metadata.pixels_per_line + padding + padding)
        / 1e9
    )

    downsampled_kymos = [
        kymo.downsampled_by(time_factor=t_factor, position_factor=p_factor),
        # Test whether sequential downsampling works out correctly as well
        kymo.downsampled_by(position_factor=p_factor).downsampled_by(time_factor=t_factor),
        kymo.downsampled_by(time_factor=t_factor).downsampled_by(position_factor=p_factor),
    ]

    for kymo_ds in downsampled_kymos:
        assert kymo_ds.name == "downsampler"
        np.testing.assert_allclose(kymo_ds.get_image("red"), ds_image)
        np.testing.assert_allclose(kymo_ds.get_image("green"), np.zeros(ds_image.shape))  # missing
        np.testing.assert_allclose(kymo_ds.start, ref.start)
        np.testing.assert_allclose(
            kymo_ds.pixelsize_um,
            np.array(ref.metadata.pixelsize_um) * p_factor,
        )
        np.testing.assert_allclose(kymo_ds.pixelsize, ref.metadata.pixelsize_um[0] * p_factor)
        np.testing.assert_allclose(kymo_ds.line_time_seconds, ref_line_time)
        assert not kymo_ds.contiguous
        with pytest.raises(
            NotImplementedError,
            match=re.escape("Per-pixel timestamps are no longer available after downsampling"),
        ):
            kymo_ds.timestamps


def test_side_no_side_effects_downsampling(downsampling_kymo):
    """Test whether downsampling doesn't have side effects on the original kymo"""
    kymo, ref = downsampling_kymo
    _ = kymo.downsampled_by(time_factor=2, position_factor=2)

    np.testing.assert_allclose(kymo.get_image("red"), ref.image[:, :, 0])
    np.testing.assert_allclose(kymo.start, ref.start)
    np.testing.assert_allclose(kymo.pixelsize_um, ref.metadata.pixelsize_um)
    np.testing.assert_allclose(kymo.line_time_seconds, ref.timestamps.line_time_seconds)
    np.testing.assert_equal(kymo.timestamps, ref.timestamps.data)
    assert kymo.contiguous
