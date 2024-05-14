import numpy as np
import pytest

from lumicks.pylake.channel import Slice, Continuous


def test_export_tiff(tmp_path, test_kymo, grab_tiff_tags):
    from os import stat

    kymo, _ = test_kymo
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


@pytest.mark.parametrize(
    "scanname, tiffname",
    [
        ("fast Y slow X", "single_frame.tiff"),
        ("fast Y slow X multiframe", "multi_frame.tiff"),
    ],
)
def test_export_tiff(
    scanname, tiffname, tmp_path, test_scans, test_scans_multiframe, grab_tiff_tags
):
    from os import stat

    test_set = test_scans | test_scans_multiframe
    scan, _ = test_set[scanname]

    filename = tmp_path / tiffname
    scan.export_tiff(filename)
    assert stat(filename).st_size > 0
    # Check if tags were properly stored, i.e. test functionality of `_tiff_image_metadata()`,
    # `_tiff_timestamp_ranges()` and `_tiff_writer_kwargs()`
    tiff_tags = grab_tiff_tags(filename)
    assert len(tiff_tags) == scan.num_frames
    ref_ts_inclusive = scan._tiff_timestamp_ranges(include_dead_time=True)
    ref_exposures = scan._tiff_timestamp_ranges(include_dead_time=False)
    for tags, timestamp_range, exposure_range in zip(tiff_tags, ref_ts_inclusive, ref_exposures):
        assert tags["ImageDescription"] == scan._tiff_image_metadata() | {
            "Exposure time (ms)": (exposure_range[1] - exposure_range[0]) * 1e-6
        }
        assert tags["DateTime"] == f"{timestamp_range[0]}:{timestamp_range[1]}"
        assert tags["Software"] == scan._tiff_writer_kwargs()["software"]
        np.testing.assert_allclose(
            tags["XResolution"][0] / tags["XResolution"][1],
            scan._tiff_writer_kwargs()["resolution"][0],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            tags["YResolution"][0] / tags["YResolution"][1],
            scan._tiff_writer_kwargs()["resolution"][1],
            rtol=1e-1,
        )
        assert tags["ResolutionUnit"] == 3  # 3 = Centimeter


def test_movie_export(tmpdir_factory, test_scans_multiframe):
    from os import stat

    tmpdir = tmpdir_factory.mktemp("pylake")

    scan, _ = test_scans_multiframe["fast Y slow X multiframe"]
    scan.export_video("red", f"{tmpdir}/red.gif", start_frame=0, stop_frame=2)
    assert stat(f"{tmpdir}/red.gif").st_size > 0
    scan.export_video("rgb", f"{tmpdir}/rgb.gif", start_frame=0, stop_frame=2)
    assert stat(f"{tmpdir}/rgb.gif").st_size > 0

    # test stop frame > num frames
    with pytest.raises(IndexError):
        scan.export_video("rgb", f"{tmpdir}/rgb.gif", start_frame=0, stop_frame=14)

    with pytest.raises(
        ValueError,
        match=(
            "channel must be 'red', 'green', 'blue' or a combination of 'r', 'g', "
            "and/or 'b', got 'gray'."
        ),
    ):
        scan.export_video("gray", "dummy.gif")  # Gray is not a color!


@pytest.mark.parametrize("vertical, channel", [(False, "red"), (True, "red"), (False, "rgb")])
def test_correlated_movie_export(tmpdir_factory, test_scans_multiframe, vertical, channel):
    from os import stat

    tmpdir = tmpdir_factory.mktemp("pylake")
    scan, _ = test_scans_multiframe["fast Y slow X multiframe"]
    corr_data = Slice(
        Continuous(np.arange(1000), scan.start, int(1e9)), labels={"title": "title", "y": "y"}
    )

    scan.export_video(
        channel,
        f"{tmpdir}/{channel}_corr.gif",
        start_frame=None,
        stop_frame=None,
        channel_slice=corr_data,
        vertical=vertical,
    )
    assert stat(f"{tmpdir}/{channel}_corr.gif").st_size > 0
