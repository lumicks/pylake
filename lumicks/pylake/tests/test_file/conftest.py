import json

import h5py
import numpy as np
import pytest

from ..data.mock_file import MockDataFile_v1, MockDataFile_v2
from ..data.mock_json import mock_force_feedback_json
from ..data.mock_confocal import generate_scan_json

# fmt: off
counts = np.uint32([2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 8, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0,
                    0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 8, 0])

infowave = np.uint8([1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 2,
                        0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2,
                        1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 2,
                        0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2])
# fmt: on


@pytest.fixture(scope="module", params=[MockDataFile_v1, MockDataFile_v2])
def h5_file(tmpdir_factory, request):
    mock_class = request.param

    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = mock_class(tmpdir.join("%s.h5" % mock_class.__class__.__name__))
    mock_file.write_metadata()

    mock_file.make_continuous_channel("Force HF", "Force 1x", 1, 10, np.arange(5.0))
    mock_file.make_continuous_channel("Force HF", "Force 1y", 1, 10, np.arange(5.0, 10.0))
    mock_file.make_continuous_channel("Force HF", "Force 1z", 1, 10, np.arange(10.0, 15.0))

    mock_file.make_timeseries_channel("Force LF", "Force 1x", [(1, 1.1), (2, 2.1)])
    mock_file.make_timeseries_channel("Force LF", "Force 1y", [(1, 1.2), (2, 2.2)])
    mock_file.make_timeseries_channel("Force LF", "Force 1z", [(1, 1.3), (2, 2.3)])

    if mock_class == MockDataFile_v2:
        mock_file.make_timetags_channel(
            "Photon Time Tags", "Red", np.arange(10, 100, step=10, dtype=np.int64)
        )

        calibration_time_field = "Stop time (ns)"
        mock_file.make_calibration_data("1", "Force 1x", {calibration_time_field: 0})
        mock_file.make_calibration_data("2", "Force 1x", {calibration_time_field: 1})
        mock_file.make_calibration_data("3", "Force 1x", {calibration_time_field: 10})
        mock_file.make_calibration_data("4", "Force 1x", {calibration_time_field: 100})
        mock_file.make_calibration_data("1", "Force 1y", {calibration_time_field: 0})
        mock_file.make_calibration_data("2", "Force 1y", {calibration_time_field: 1})
        mock_file.make_calibration_data("3", "Force 1y", {calibration_time_field: 10})
        mock_file.make_calibration_data("4", "Force 1y", {calibration_time_field: 100})
        mock_file.make_calibration_data("1", "Force 1z", {calibration_time_field: 0})
        mock_file.make_calibration_data("2", "Force 1z", {calibration_time_field: 1})
        mock_file.make_calibration_data("3", "Force 1z", {calibration_time_field: 10})
        mock_file.make_calibration_data("4", "Force 1z", {calibration_time_field: 100})

        mock_file.make_marker("test_marker", {"Start time (ns)": 100, "Stop time (ns)": 200})
        mock_file.make_marker("test_marker2", {"Start time (ns)": 200, "Stop time (ns)": 300})
        mock_file.make_marker(
            "force feedback",
            {"Start time (ns)": 200, "Stop time (ns)": 300},
            payload=mock_force_feedback_json(),
        )
        mock_file.make_note(
            "test_note", {"Start time (ns)": 100, "Stop time (ns)": 100}, "Note content"
        )
        mock_file.make_fd()

        json_kymo = generate_scan_json([{"axis": 0, "num of pixels": 5, "pixel size (nm)": 10.0}])

        # Generate lines at 1 Hz
        freq = 1e9 / 16
        mock_file.make_continuous_channel("Photon count", "Red", np.int64(20e9), freq, counts)
        mock_file.make_continuous_channel("Photon count", "Green", np.int64(20e9), freq, counts)
        mock_file.make_continuous_channel("Photon count", "Blue", np.int64(20e9), freq, counts)
        mock_file.make_continuous_channel("Info wave", "Info wave", np.int64(20e9), freq, infowave)

        ds = mock_file.make_json_data("Kymograph", "Kymo1", json_kymo)
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

        # Force channel that overlaps kymo; step from high to low force
        # We want two lines of the kymo to have a force of 30, the other 10. Force starts 5 samples
        # before the kymograph. First kymotrack is 15 samples long, second is 16 samples long, which
        # means the third line starts after 31 + 5 = 36 samples
        force_data = np.hstack((np.ones(37) * 30, np.ones(33) * 10))
        force_start = np.int64(ds.attrs["Start time (ns)"] - (freq * 5))  # before infowave
        mock_file.make_continuous_channel("Force HF", "Force 2x", force_start, freq, force_data)
        mock_file.make_calibration_data("1", "Force 2x", {calibration_time_field: 0})
        mock_file.make_calibration_data("2", "Force 2x", {calibration_time_field: 1})
        mock_file.make_calibration_data("3", "Force 2x", {calibration_time_field: 10})
        mock_file.make_calibration_data("4", "Force 2x", {calibration_time_field: 100})

        # Single frame image
        json = generate_scan_json(
            [
                {"axis": 1, "num of pixels": 4, "pixel size (nm)": 191.0},
                {"axis": 0, "num of pixels": 5, "pixel size (nm)": 197.0},
            ],
        )
        ds = mock_file.make_json_data("Scan", "fast Y slow X", json)
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

        # Multi frame image
        json = generate_scan_json(
            [
                {"axis": 1, "num of pixels": 4, "pixel size (nm)": 191.0},
                {"axis": 0, "num of pixels": 3, "pixel size (nm)": 197.0},
            ],
        )
        ds = mock_file.make_json_data("Scan", "fast Y slow X multiframe", json)
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

        # Multiframe frame image
        json = generate_scan_json(
            [
                {"axis": 0, "num of pixels": 4, "pixel size (nm)": 191.0},
                {"axis": 2, "num of pixels": 3, "pixel size (nm)": 197.0},
            ],
        )
        ds = mock_file.make_json_data("Scan", "fast X slow Z multiframe", json)
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

        # Multiframe frame image
        json = generate_scan_json(
            [
                {"axis": 1, "num of pixels": 4, "pixel size (nm)": 191.0},
                {"axis": 2, "num of pixels": 3, "pixel size (nm)": 197.0},
            ],
        )
        ds = mock_file.make_json_data("Scan", "fast Y slow Z multiframe", json)
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

        # Point Scan
        ps_json_string = generate_scan_json([])
        ds = mock_file.make_json_data("Point Scan", "PointScan1", ps_json_string)
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

    return mock_file.file


@pytest.fixture(scope="module", params=[MockDataFile_v1, MockDataFile_v2])
def h5_file_missing_meta(tmpdir_factory, request):
    mock_class = request.param

    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = mock_class(tmpdir.join("%s.h5" % mock_class.__class__.__name__))
    mock_file.write_metadata()

    if mock_class == MockDataFile_v2:
        enc = json.JSONEncoder()

        # Generate lines at 1 Hz
        freq = 1e9 / 16

        # Single frame image - NO metadata
        ds = mock_file.make_json_data("Scan", "fast Y slow X no meta", enc.encode({}))
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

        # Single frame image - ok loading
        ds = mock_file.make_json_data(
            "Scan",
            "fast Y slow X",
            generate_scan_json(
                [
                    {"axis": 1, "num of pixels": 4, "pixel size (nm)": 191.0},
                    {"axis": 0, "num of pixels": 5, "pixel size (nm)": 197.0},
                ]
            ),
        )
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

    return mock_file.file


@pytest.fixture(scope="module")
def h5_file_invalid_version(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("pylake")

    mock_file = h5py.File(tmpdir.join("invalid.h5"), "w")
    mock_file.attrs["Bluelake version"] = "unknown"
    mock_file.attrs["File format version"] = 254

    return mock_file


@pytest.fixture(scope="module", params=[MockDataFile_v2])
def h5_kymo_as_scan(tmpdir_factory, request):
    mock_class = request.param

    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = mock_class(tmpdir.join("%s.h5" % mock_class.__class__.__name__))
    mock_file.write_metadata()

    json_kymo = generate_scan_json([{"axis": 1, "num of pixels": 4, "pixel size (nm)": 191.0}])
    ds = mock_file.make_json_data("Scan", "Kymo1", json_kymo)
    ds.attrs["Start time (ns)"] = np.int64(20e9)
    ds.attrs["Stop time (ns)"] = np.int64(100e9)

    return mock_file.file


@pytest.fixture(scope="module", params=[MockDataFile_v2])
def h5_custom_detectors(tmpdir_factory, request):
    mock_class = request.param

    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = mock_class(tmpdir.join("%s.h5" % mock_class.__class__.__name__))
    mock_file.write_metadata()

    freq = 1e9 / 16
    mock_file.make_continuous_channel("Photon count", "Detector 1", np.int64(20e9), freq, counts)
    mock_file.make_continuous_channel("Photon count", "Detector 2", np.int64(20e9), freq, counts)
    mock_file.make_continuous_channel("Photon count", "Detector 3", np.int64(20e9), freq, counts)
    mock_file.make_continuous_channel("Info wave", "Info wave", np.int64(20e9), freq, infowave)
    return mock_file.file


@pytest.fixture(scope="module", params=[MockDataFile_v2])
def h5_two_colors(tmpdir_factory, request):
    mock_class = request.param

    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = mock_class(tmpdir.join("%s.h5" % mock_class.__class__.__name__))
    mock_file.write_metadata()

    freq = 1e9 / 16
    mock_file.make_continuous_channel("Photon count", "Red", np.int64(20e9), freq, counts)
    mock_file.make_continuous_channel("Photon count", "Blue", np.int64(20e9), freq, counts)
    mock_file.make_continuous_channel("Info wave", "Info wave", np.int64(20e9), freq, infowave)
    return mock_file.file
