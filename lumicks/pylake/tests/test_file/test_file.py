import time
from textwrap import dedent

import numpy as np
import pytest

from lumicks import pylake
from lumicks.pylake.detail.h5_helper import write_h5
from lumicks.pylake.tests.data.mock_file import MockDataFile_v2
from lumicks.pylake.tests.data.mock_confocal import generate_scan_json, generate_image_data

from . import test_file_items


def test_attributes(h5_file):
    f = pylake.File.from_h5py(h5_file)

    assert type(f.bluelake_version) is str
    assert f.format_version in [1, 2]
    assert type(f.experiment) is str
    assert type(f.description) is str
    assert type(f.guid) is str
    assert np.issubdtype(f.export_time, np.dtype(int).type)


def test_properties(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 1:
        assert f.kymos == {}
    else:
        assert len(f.kymos) == 1
    if f.format_version == 1:
        assert f.scans == {}
    else:
        assert len(f.scans) == 4
    if f.format_version == 1:
        assert f.point_scans == {}
    else:
        assert len(f.point_scans) == 1
    assert f.fdcurves == {}


def test_groups(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 1:
        assert str(f["Force HF"]) == "{'Force 1x', 'Force 1y', 'Force 1z'}"

        for x in range(0, 2):
            t = [name for name in f]
            assert set(t) == set(f)

        for x in range(0, 2):
            t = [name for name in f["Force HF"]]
            assert set(t) == set(["Force 1x", "Force 1y", "Force 1z"])


def test_contains(h5_file):
    f = pylake.File.from_h5py(h5_file)
    assert "Force HF" in f
    assert "Force 1x" in f["Force HF"]
    assert "Force HF/Force 1x" in f
    assert "Force HF" not in f["Force HF"]


def test_redirect_list(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        with pytest.warns(FutureWarning):
            f["Calibration"]

        assert f["Marker"]["test_marker"].start == 100
        assert str(type(f["FD Curve"])) == r"<class 'lumicks.pylake.group.Group'>"
        assert f["Kymograph"]["Kymo1"].start == np.int64(20e9)
        assert f["Scan"]["fast Y slow Z multiframe"].start == np.int64(20e9)
        assert f["Point Scan"]["PointScan1"].start == np.int64(20e9)


def test_repr_and_str(h5_file):
    f = pylake.File.from_h5py(h5_file)

    assert repr(f) == f"lumicks.pylake.File('{h5_file.filename}')"
    if f.format_version == 1:
        assert str(f) == dedent(
            """\
            File root metadata:
            - Bluelake version: unknown
            - Description: test
            - Experiment: test
            - Export time (ns): -1
            - File format version: 1
            - GUID: invalid

            Force HF:
              Force 1x:
              - Data type: float64
              - Size: 5
              Force 1y:
              - Data type: float64
              - Size: 5
              Force 1z:
              - Data type: float64
              - Size: 5
            Force LF:
              Force 1x:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2
              Force 1y:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2
              Force 1z:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2

            .force1x
            .force1y
            .force1z

            .downsampled_force1x
            .downsampled_force1y
            .downsampled_force1z
        """
        )
    if f.format_version == 2:
        assert str(f) == dedent(
            """\
            File root metadata:
            - Bluelake version: unknown
            - Description: test
            - Experiment: test
            - Export time (ns): -1
            - File format version: 2
            - GUID: invalid

            Force HF:
              Force 1x:
              - Data type: float64
              - Size: 5
              Force 1y:
              - Data type: float64
              - Size: 5
              Force 1z:
              - Data type: float64
              - Size: 5
              Force 2x:
              - Data type: float64
              - Size: 70
            Force LF:
              Force 1x:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2
              Force 1y:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2
              Force 1z:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2
            Info wave:
              Info wave:
              - Data type: uint8
              - Size: 64
            Photon Time Tags:
              Red:
              - Data type: int64
              - Size: 9
            Photon count:
              Blue:
              - Data type: uint32
              - Size: 64
              Green:
              - Data type: uint32
              - Size: 64
              Red:
              - Data type: uint32
              - Size: 64

            .markers
              - force feedback
              - test_marker
              - test_marker2

            .kymos
              - Kymo1

            .scans
              - fast X slow Z multiframe
              - fast Y slow X
              - fast Y slow X multiframe
              - fast Y slow Z multiframe

            .notes
              - test_note

            .point_scans
              - PointScan1

            .force1x
              .calibration
            .force1y
              .calibration
            .force1z
              .calibration
            .force2x
              .calibration

            .downsampled_force1x
              .calibration
            .downsampled_force1y
              .calibration
            .downsampled_force1z
              .calibration
        """
        )


def test_invalid_file_format(h5_file_invalid_version):
    with pytest.raises(Exception):
        f = pylake.File.from_h5py(h5_file_invalid_version)


def test_missing_metadata(h5_file_missing_meta):
    f = pylake.File.from_h5py(h5_file_missing_meta)
    if f.format_version == 2:
        with pytest.warns(
            UserWarning,
            match="Scan 'fast Y slow X no meta' is missing metadata and cannot be loaded",
        ):
            scans = f.scans
            assert len(scans) == 1


def _internal_h5_export_api(file, *args, **kwargs):
    return write_h5(file, *args, **kwargs)


def _public_h5_export_api(file, *args, **kwargs):
    return file.save_as(*args, **kwargs, verbose=False)


@pytest.mark.parametrize("save_h5", [_internal_h5_export_api, _public_h5_export_api])
def test_h5_export(tmpdir_factory, h5_file, save_h5):
    f = pylake.File.from_h5py(h5_file)
    tmpdir = tmpdir_factory.mktemp("pylake")

    new_file = f"{tmpdir}/copy.h5"
    save_h5(f, new_file, 5)
    g = pylake.File(new_file)
    assert str(g) == str(f)

    # Verify that all attributes are there and correct
    test_file_items.test_scans(g.h5)
    test_file_items.test_kymos(g.h5)
    test_attributes(g.h5)
    test_file_items.test_channels(g.h5)
    test_file_items.test_calibration(g.h5)
    test_file_items.test_marker(g.h5)
    test_properties(g.h5)

    new_file = f"{tmpdir}/omit_LF1y.h5"
    save_h5(f, new_file, 5, omit_data={"Force LF/Force 1y"})
    omit_lf1y = pylake.File(new_file)

    np.testing.assert_allclose(
        omit_lf1y["Force LF"]["Force 1x"].data, f["Force LF"]["Force 1x"].data
    )
    np.testing.assert_allclose(
        omit_lf1y["Force HF"]["Force 1x"].data, f["Force HF"]["Force 1x"].data
    )
    np.testing.assert_allclose(
        omit_lf1y["Force HF"]["Force 1y"].data, f["Force HF"]["Force 1y"].data
    )
    with pytest.raises(KeyError):
        assert np.any(omit_lf1y["Force LF"]["Force 1y"].data)

    for ix, drop_style in enumerate(
        ({"*/Force 1y"}, ["*/Force 1y"], ("*/Force 1y",), "*/Force 1y")
    ):
        new_file = f"{tmpdir}/omit_1y_{ix}.h5"
        save_h5(f, new_file, 5, omit_data=drop_style)
        omit_1y = pylake.File(new_file)

        np.testing.assert_allclose(
            omit_1y["Force LF"]["Force 1x"].data, f["Force LF"]["Force 1x"].data
        )
        np.testing.assert_allclose(
            omit_1y["Force HF"]["Force 1x"].data, f["Force HF"]["Force 1x"].data
        )
        with pytest.raises(KeyError):
            np.testing.assert_allclose(
                omit_1y["Force HF"]["Force 1y"].data, f["Force HF"]["Force 1y"].data
            )
        with pytest.raises(KeyError):
            assert np.any(omit_1y["Force LF"]["Force 1y"].data)

    new_file = f"{tmpdir}/omit_hf.h5"
    save_h5(f, new_file, 5, omit_data={"Force HF/*"})
    omit_hf = pylake.File(new_file)

    np.testing.assert_allclose(omit_hf["Force LF"]["Force 1x"].data, f["Force LF"]["Force 1x"].data)
    np.testing.assert_allclose(omit_hf["Force LF"]["Force 1y"].data, f["Force LF"]["Force 1y"].data)
    with pytest.raises(KeyError):
        np.testing.assert_allclose(
            omit_hf["Force HF"]["Force 1x"].data, f["Force HF"]["Force 1x"].data
        )
    with pytest.raises(KeyError):
        np.testing.assert_allclose(
            omit_hf["Force HF"]["Force 1y"].data, f["Force HF"]["Force 1y"].data
        )

    new_file = f"{tmpdir}/omit_two.h5"
    save_h5(f, new_file, 5, omit_data={"Force HF/*", "*/Force 1y"})
    omit_two = pylake.File(new_file)

    np.testing.assert_allclose(
        omit_two["Force LF"]["Force 1x"].data, f["Force LF"]["Force 1x"].data
    )
    with pytest.raises(KeyError):
        np.testing.assert_allclose(
            omit_two["Force LF"]["Force 1y"].data, f["Force LF"]["Force 1y"].data
        )
    with pytest.raises(KeyError):
        np.testing.assert_allclose(
            omit_two["Force HF"]["Force 1x"].data, f["Force HF"]["Force 1x"].data
        )
    with pytest.raises(KeyError):
        np.testing.assert_allclose(
            omit_two["Force HF"]["Force 1y"].data, f["Force HF"]["Force 1y"].data
        )


def test_timeseries_performance(tmpdir_factory):
    # This is a regression test for a bug that showed catastrophic performance for TimeSeries data.
    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = MockDataFile_v2(tmpdir.join("regression_test_timeseries.h5"))
    mock_file.write_metadata()

    ts = np.arange(int(1e10), int(10e10), int(1e7), dtype=np.int64)
    data = [datum for datum in zip(ts, np.arange(1e10, 10e10, 1e7))]
    mock_file.make_timeseries_channel("test", "test", data)

    f = pylake.File.from_h5py(mock_file.file)
    tic = time.time()
    np.testing.assert_allclose(f["test"]["test"].timestamps, ts)
    assert time.time() - tic < 0.1, "Grabbing timestamps from TimeSeries is too slow"


@pytest.mark.parametrize("save_h5", [_internal_h5_export_api, _public_h5_export_api])
def test_h5_cropped_export_channels(tmpdir_factory, save_h5):
    start = int(1e9)
    dt = int(1e9 / 78125)
    calibration_points = [start - dt, start, start + dt, start + 15 * dt]

    mock_class = MockDataFile_v2
    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = mock_class(tmpdir.join(f"channels_to_crop.h5"))
    mock_file.write_metadata()
    mock_file.make_continuous_channel("Force HF", "Force 1x", start, dt, np.arange(10.0))
    for ix, t in enumerate(calibration_points):
        mock_file.make_calibration_data(str(ix), "Force 1x", {"Stop time (ns)": t, "s": str(ix)})
    timestamps = np.arange(start, start + dt * 10, dt, dtype=np.int64)
    data = [(t, d) for t, d in zip(timestamps, np.arange(0.0, 10.0, dtype=float))]
    mock_file.make_timeseries_channel("Low Freq", "Yes", data)
    mock_file.make_timetags_channel("TimeTags", "They Exist?", timestamps)
    lk_file = pylake.File.from_h5py(mock_file.file)

    def crop_h5(name, crop):
        target_file = tmpdir.join(f"{name}_cropped.h5")
        lk_file.save_as(target_file, crop_time_range=crop, verbose=False)
        return pylake.File(target_file)

    ref_tags = np.arange(start, start + 10 * dt, dt, dtype=np.int64)
    for ix, (crop, result, ref_tags, ref_calib) in enumerate(
        (
            ((start + 5 * dt, start + 100 * dt), np.arange(5.0, 10.0), ref_tags[5:10], [2]),
            ((start + 5 * dt, start + 8 * dt), np.arange(5.0, 8.0), ref_tags[5:8], [2]),
            ((0, start + 8 * dt), np.arange(0.0, 8.0), ref_tags[:8], [1, 2]),
            ((start + dt, start + 8 * dt), np.arange(1.0, 8.0), ref_tags[1:8], [2]),
            ((start + 11 * dt, start + 100 * dt), None, None, None),
            ((start - 11 * dt, start), None, None, None),
        )
    ):
        f = crop_h5(f"channels_{ix}", crop=crop)
        if result is not None:
            np.testing.assert_allclose(f.force1x.data, result)
            for c, c_ref in zip(f.force1x.calibration, ref_calib):
                assert c["s"] == str(c_ref)

            np.testing.assert_allclose(f.force1x.timestamps, ref_tags)
            np.testing.assert_allclose(f["Low Freq"]["Yes"].data, result)
            np.testing.assert_allclose(f["Low Freq"]["Yes"].timestamps, ref_tags)
            np.testing.assert_allclose(f["TimeTags"]["They Exist?"].timestamps, ref_tags)
        else:
            # Channels will not be there if they are sliced off
            assert not f.force1x
            assert "Yes" not in f["Low Freq"]
            assert "They Exist?" not in f["TimeTags"]


def test_h5_cropped_export_confocal(tmpdir_factory):
    start = 1689369419 * int(1e9)
    dt = int(1e9 / 78125)

    image = np.eye(5)
    infowave, photon_counts = generate_image_data(image, 5, 2)
    infowave, photon_counts = (np.tile(arr, (8,)) for arr in (infowave, photon_counts[0]))

    json_scan = generate_scan_json(
        [
            {"axis": axis, "num of pixels": num_pixels, "pixel size (nm)": pixel_size}
            for pixel_size, axis, num_pixels in zip([1, 1], [0, 1], image.shape)
        ]
    )
    json_kymo = generate_scan_json(
        [{"axis": 0, "num of pixels": image.shape[0], "pixel size (nm)": 1.0}]
    )
    json_point = generate_scan_json([])

    stop = start + len(infowave) * dt
    mock_class = MockDataFile_v2
    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = mock_class(tmpdir.join(f"channels_to_crop.h5"))
    mock_file.write_metadata()
    mock_file.make_continuous_channel("Photon count", "Red", start, dt, photon_counts)
    mock_file.make_continuous_channel("Photon count", "Green", start, dt, photon_counts)
    mock_file.make_continuous_channel("Photon count", "Blue", start, dt, photon_counts)
    mock_file.make_continuous_channel("Info wave", "Info wave", start, dt, infowave)
    ds1 = mock_file.make_json_data("Scan", "Scan1", json_scan)
    ds2 = mock_file.make_json_data("Kymograph", "Kymo1", json_kymo)
    ds3 = mock_file.make_json_data("Point Scan", "PointScan1", json_point)
    for ds in (ds1, ds2, ds3):
        ds.attrs["Start time (ns)"] = np.int64(start)
        ds.attrs["Stop time (ns)"] = np.int64(start + len(infowave) * dt)

    lk_file = pylake.File.from_h5py(mock_file.file)

    def cropped_h5(name, crop=None):
        target_file = tmpdir.join(f"{name}_cropped.h5")
        lk_file.save_as(target_file, crop_time_range=crop, verbose=False)
        return pylake.File(target_file)

    for ix, crop in enumerate(
        (
            (start, stop + 1),  # No change
            (start + (stop - start) // 2, stop + 1),  # Middle to end
            (start, start + (stop - start) // 2),  # Start to middle
            (start - 1000, start - 1),  # Before it even begins (no data)
            (stop + 1000, stop + (stop - start)),  # After it ends (no data)
            (start, start + 1),
            (start, start),
        )
    ):
        f = cropped_h5(f"channels_{ix}", crop=crop)
        ref_image = lk_file["Scan"]["Scan1"][crop[0] : crop[1]].get_image("red")
        if ref_image.size:
            np.testing.assert_allclose(f["Scan"]["Scan1"].get_image("red"), ref_image)
        else:
            assert "Scan1" not in f["Scan"]

        ref_image = lk_file["Kymograph"]["Kymo1"][crop[0] : crop[1]].get_image("red")
        if ref_image.size:
            np.testing.assert_allclose(f["Kymograph"]["Kymo1"].get_image("red"), ref_image)
        else:
            assert "Kymo1" not in f["Kymograph"]

        ref_data = lk_file["Point Scan"]["PointScan1"][crop[0] : crop[1]].red_photon_count
        if ref_data:
            np.testing.assert_allclose(f["Point Scan"]["PointScan1"].red_photon_count, ref_data)
        else:
            assert (
                "PointScan1" not in f["Point Scan"]
                or not f["Point Scan"]["PointScan1"].red_photon_count
            )


def test_detector_mapping(h5_custom_detectors, h5_two_colors):
    custom_correct = pylake.File.from_h5py(
        h5_custom_detectors,
        rgb_to_detectors={"Red": "Detector 1", "Green": "Detector 2", "Blue": "Detector 3"},
    )
    assert custom_correct.red_photon_count.data.size
    assert custom_correct.green_photon_count.data.size
    assert custom_correct.blue_photon_count.data.size

    # None of these exist, since this is a custom photon count file
    with pytest.warns(RuntimeWarning, match="Invalid RGB to detector mapping"):
        custom_empty = pylake.File.from_h5py(
            h5_custom_detectors, rgb_to_detectors={"Red": "Red", "Green": "Green", "Blue": "Blue"}
        )
        assert not custom_empty.red_photon_count.data.size
        assert not custom_empty.green_photon_count.data.size
        assert not custom_empty.blue_photon_count.data.size

    # Green photon count channel doesn't exist, so we issue a warning about this
    with pytest.warns(RuntimeWarning, match="Invalid RGB to detector mapping"):
        two_color = pylake.File.from_h5py(
            h5_two_colors, rgb_to_detectors={"Red": "Red", "Green": "Green", "Blue": "Blue"}
        )
        assert two_color.red_photon_count.data.size
        assert not two_color.green_photon_count.data.size
        assert two_color.blue_photon_count.data.size

    # Green photon count channel doesn't exist, _but_ we are using defaults, so we do not explicitly
    # issue a warning (this is the old reference behavior). People may not even want to show
    # images, so it's weird to bother them with a warning.
    two_color_default = pylake.File.from_h5py(h5_two_colors)
    assert two_color_default.red_photon_count.data.size
    assert not two_color_default.green_photon_count.data.size
    assert two_color_default.blue_photon_count.data.size

    # Valid mapping
    double_red = pylake.File.from_h5py(
        h5_two_colors, rgb_to_detectors={"Red": "Red", "Green": "Red", "Blue": "Blue"}
    )
    np.testing.assert_allclose(double_red.green_photon_count.data, two_color.red_photon_count.data)
    assert double_red.blue_photon_count.data.size

    # Explicitly setting `"None"` doesn't issue a warning
    no_green = pylake.File.from_h5py(
        h5_two_colors, rgb_to_detectors={"Red": "Red", "Green": "None", "Blue": "Blue"}
    )
    assert no_green.red_photon_count.data.size
    assert not no_green.green_photon_count.data.size
    assert no_green.blue_photon_count.data.size


@pytest.mark.parametrize(
    "mapping",
    [
        {"red": "Red", "green": "Red", "blue": "Blue"},  # Incorrect capitalization
        {"Red": "Red", "Magenta": "Red", "Blue": "Blue"},  # Wrong name
    ],
)
def test_detector_mapping_invalid_color(h5_two_colors, mapping):
    with pytest.raises(ValueError, match="Invalid color mapping"):
        pylake.File.from_h5py(h5_two_colors, rgb_to_detectors=mapping)
