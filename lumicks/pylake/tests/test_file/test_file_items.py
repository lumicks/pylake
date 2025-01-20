import numpy as np
import pytest

from lumicks import pylake
from lumicks.pylake.calibration import ForceCalibrationList

from ..data.mock_json import force_feedback_dict


def test_channels(h5_file):
    f = pylake.File.from_h5py(h5_file)

    np.testing.assert_allclose(f.force1x.data, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(f.force1x.timestamps, [1, 11, 21, 31, 41])
    assert "x" not in f.force1x.labels
    assert f.force1x.labels["y"] == "Force (pN)"
    assert f.force1x.labels["title"] == "Force HF/Force 1x"

    np.testing.assert_allclose(f.downsampled_force1x.data, [1.1, 2.1])
    np.testing.assert_allclose(f.downsampled_force1x.timestamps, [1, 2])
    assert "x" not in f.downsampled_force1x.labels
    assert f.downsampled_force1x.labels["y"] == "Force (pN)"
    assert f.downsampled_force1x.labels["title"] == "Force LF/Force 1x"

    vsum_force = np.sqrt(f.downsampled_force1x.data**2 + f.downsampled_force1y.data**2)
    np.testing.assert_allclose(f.downsampled_force1.data, vsum_force)
    np.testing.assert_allclose(f.downsampled_force1.timestamps, [1, 2])
    assert "x" not in f.downsampled_force1.labels
    assert f.downsampled_force1.labels["y"] == "Force (pN)"
    assert f.downsampled_force1.labels["title"] == "Force LF/Force 1"


def test_calibration(h5_file):
    f = pylake.File.from_h5py(h5_file)

    assert type(f.force1x.calibration) is ForceCalibrationList or list
    assert type(f.downsampled_force1.calibration) is ForceCalibrationList or list
    assert type(f.downsampled_force1x.calibration) is ForceCalibrationList or list

    if f.format_version == 1:
        # v1 version doesn't include calibration data field
        assert len(f.force1x.calibration) == 0
        assert len(f.downsampled_force1.calibration) == 0
        assert len(f.downsampled_force1x.calibration) == 0
        assert len(f["Force HF"]["Force 1x"].calibration) == 0
        assert len(f["Force LF"]["Force 1x"].calibration) == 0

    if f.format_version == 2:
        assert len(f.force1x.calibration) == 3
        assert len(f.downsampled_force1.calibration) == 0
        assert len(f.downsampled_force1x.calibration) == 1
        assert len(f["Force HF"]["Force 1x"].calibration) == 3
        assert len(f["Force LF"]["Force 1x"].calibration) == 1

        assert f["Force HF"]["Force 1x"].calibration == f.force1x.calibration
        assert f["Force HF"]["Force 2x"].calibration == f.force2x.calibration
        assert f.force1x.calibration[0].applied_at == 1
        assert f.force1x.calibration[1].applied_at == 3
        assert f.force1x.calibration[2].applied_at == 10
        assert f.force1y.calibration[0].applied_at == 1
        assert f.force1y.calibration[1].applied_at == 3
        assert f.force1y.calibration[2].applied_at == 8
        assert f.force1y.calibration[3].applied_at == 10
        assert f.force2x.calibration[0].applied_at == 100
        assert not f.force1x.calibration[0].has_data
        assert len(f.force1x.calibration[0].voltage.data) == 0
        assert len(f.force1x.calibration[0].sum_voltage.data) == 0
        assert len(f.force1x.calibration[0].driving.data) == 0

        # Verify that one specific calibration has data associated with it
        idx = 2
        assert not f.force1x.calibration[idx].has_data
        np.testing.assert_equal(f.force1y.calibration[idx].sum_voltage.data, np.arange(5.0))
        assert f.force1y.calibration[idx].sum_voltage.sample_rate == 78125
        assert f.force1y.calibration[idx].sum_voltage.labels["x"] == "Time (s)"
        assert f.force1y.calibration[idx].sum_voltage.labels["y"] == "Sum voltage (V)"
        assert f.force1y.calibration[idx].sum_voltage.labels["title"] == "Sum voltage 1"

        np.testing.assert_equal(f.force1y.calibration[idx].voltage.data, np.arange(15.0))
        assert f.force1y.calibration[idx].voltage.sample_rate == 78125
        assert f.force1y.calibration[idx].voltage.labels["x"] == "Time (s)"
        assert f.force1y.calibration[idx].voltage.labels["y"] == "Uncalibrated Force (V)"
        assert f.force1y.calibration[idx].voltage.labels["title"] == "Uncalibrated Force 1y"

        np.testing.assert_equal(f.force1y.calibration[idx].driving.data, np.arange(25.0))
        assert f.force1y.calibration[idx].driving.sample_rate == 78125
        assert f.force1y.calibration[idx].driving.labels["x"] == "Time (s)"
        assert f.force1y.calibration[idx].driving.labels["y"] == r"Driving data ($\mu$m)"
        assert f.force1y.calibration[idx].driving.labels["title"] == "Driving data for axis y"


def test_calibration_lazy(monkeypatch, h5_file):
    import h5py

    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        data_accessed = False
        h5_array = h5py.Dataset.__array__

        def patched_array(self, *args, **kwargs):
            nonlocal data_accessed
            data_accessed = True
            return h5_array(self, *args, **kwargs)

        with monkeypatch.context() as m:
            m.setattr("h5py.Dataset.__array__", patched_array)

            assert not f.force1x.calibration[2].has_data
            assert f.force1y.calibration[2].voltage.sample_rate == 78125
            assert not data_accessed  # Verify that we didn't read the slice yet

            # Read the data
            _ = f.force1y.calibration[2].voltage.data

            assert data_accessed


def test_marker(h5_file):
    f = pylake.File.from_h5py(h5_file)

    if f.format_version == 2:
        assert np.isclose(f["Marker/test_marker"].start, 100)
        assert np.isclose(f["Marker"]["test_marker"].start, 100)

        assert np.isclose(f.markers["test_marker"].start, 100)
        assert np.isclose(f.markers["test_marker"].stop, 200)
        assert np.isclose(f.markers["test_marker2"].start, 200)
        assert np.isclose(f.markers["test_marker2"].stop, 300)


def test_marker_metadata(h5_file):
    f = pylake.File.from_h5py(h5_file)

    if f.format_version == 2:
        assert not f.markers["test_marker"]._json
        assert not f.markers["test_marker2"]._json
        assert np.isclose(f.markers["force feedback"].start, 200)
        assert np.isclose(f.markers["force feedback"].stop, 300)
        assert f.markers["force feedback"]._json == force_feedback_dict


def test_scans(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        scan = f.scans["fast Y slow X"]
        assert scan.pixels_per_line == 4  # Fast axis
        np.testing.assert_allclose(
            scan.get_image("red"),
            np.transpose([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0]]),
        )

        scan2 = f.scans["fast Y slow X multiframe"]
        reference = np.array(
            [[[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0]], [[0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]]]
        )
        reference = np.transpose(reference, [0, 2, 1])
        np.testing.assert_allclose(scan2.get_image("red"), reference)

        scan2 = f.scans["fast Y slow X multiframe"]
        rgb = np.zeros((2, 4, 3, 3))
        rgb[:, :, :, 0] = reference
        rgb[:, :, :, 1] = reference
        rgb[:, :, :, 2] = reference
        np.testing.assert_allclose(scan2.get_image("rgb"), rgb)


def test_kymos(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]
        assert kymo.pixels_per_line == 5
        np.testing.assert_allclose(
            kymo.get_image("red"),
            np.transpose([[2, 0, 0, 0, 2], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]]),
        )

        np.testing.assert_allclose(kymo.red_power.timestamps, [np.int64(15e9), np.int64(20e9)])
        np.testing.assert_allclose(kymo.red_power.data, [10, 15])
        np.testing.assert_allclose(kymo.green_power.timestamps, [np.int64(20e9)])
        np.testing.assert_allclose(kymo.green_power.data, [15])
        np.testing.assert_allclose(kymo.blue_power.timestamps, [])
        np.testing.assert_allclose(kymo.blue_power.data, [])


def test_notes(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        name = "test_note"
        note = f.notes[name]
        assert note.name == name
        assert note.text == "Note content"
        assert note.start == 100
        assert note.stop == 100


def test_kymos_in_scans(h5_kymo_as_scan):
    """Tests whether Kymos accidentally put in Scan are loaded correctly"""
    f = pylake.File.from_h5py(h5_kymo_as_scan)

    assert len(f.kymos) == 1
    assert f.kymos["Kymo1"].name == "Kymo1"

    with pytest.warns():
        scan_dict = f.scans
        assert not scan_dict
