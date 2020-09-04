import numpy as np
from lumicks import pylake
import pytest
from textwrap import dedent


def test_scans(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        scan = f.scans["fast Y slow X"]
        assert scan.pixels_per_line == 4  # Fast axis
        assert np.allclose(scan.red_image, np.transpose([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0]]))

        scan2 = f.scans["fast Y slow X multiframe"]
        reference = np.array([[[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0]], [[0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]]])
        reference = np.transpose(reference, [0, 2, 1])
        assert np.allclose(scan2.red_image, reference)

        scan2 = f.scans["fast Y slow X multiframe"]
        rgb = np.zeros((2, 4, 3, 3))
        rgb[:, :, :, 0] = reference
        rgb[:, :, :, 1] = reference
        rgb[:, :, :, 2] = reference
        assert np.allclose(scan2.rgb_image, rgb)


def test_kymos(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]
        assert kymo.pixels_per_line == 5
        assert np.allclose(kymo.red_image, np.transpose([[2, 0, 0, 0, 2], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]]))


def test_attributes(h5_file):
    f = pylake.File.from_h5py(h5_file)

    assert type(f.bluelake_version) is str
    assert f.format_version in [1, 2]
    assert type(f.experiment) is str
    assert type(f.description) is str
    assert type(f.guid) is str
    assert np.issubdtype(f.export_time, int)


def test_channels(h5_file):
    f = pylake.File.from_h5py(h5_file)

    assert np.allclose(f.force1x.data, [0, 1, 2, 3, 4])
    assert np.allclose(f.force1x.timestamps, [1, 11, 21, 31, 41])
    assert "x" not in f.force1x.labels
    assert f.force1x.labels["y"] == "Force (pN)"
    assert f.force1x.labels["title"] == "Force HF/Force 1x"

    assert np.allclose(f.downsampled_force1x.data, [1.1, 2.1])
    assert np.allclose(f.downsampled_force1x.timestamps, [1, 2])
    assert "x" not in f.downsampled_force1x.labels
    assert f.downsampled_force1x.labels["y"] == "Force (pN)"
    assert f.downsampled_force1x.labels["title"] == "Force LF/Force 1x"

    vsum_force = np.sqrt(f.downsampled_force1x.data**2 + f.downsampled_force1y.data**2)
    assert np.allclose(f.downsampled_force1.data, vsum_force)
    assert np.allclose(f.downsampled_force1.timestamps, [1, 2])
    assert "x" not in f.downsampled_force1.labels
    assert f.downsampled_force1.labels["y"] == "Force (pN)"
    assert f.downsampled_force1.labels["title"] == "Force LF/Force 1"


def test_calibration(h5_file):
    f = pylake.File.from_h5py(h5_file)

    assert type(f.force1x.calibration) is list
    assert type(f.downsampled_force1.calibration) is list
    assert type(f.downsampled_force1x.calibration) is list

    if f.format_version == 1:
        # v1 version doesn't include calibration data field
        assert len(f.force1x.calibration) == 0
        assert len(f.downsampled_force1.calibration) == 0
        assert len(f.downsampled_force1x.calibration) == 0

    if f.format_version == 2:
        assert len(f.force1x.calibration) == 2
        assert len(f.downsampled_force1.calibration) == 0
        assert len(f.downsampled_force1x.calibration) == 1


def test_marker(h5_file):
    f = pylake.File.from_h5py(h5_file)

    if f.format_version == 2:
        with pytest.warns(FutureWarning):
            m = f["Marker"]

        with pytest.raises(IndexError):
            m["test_marker"]

        assert np.isclose(f.markers["test_marker"].start, 100)
        assert np.isclose(f.markers["test_marker"].stop, 200)
        assert np.isclose(f.markers["test_marker2"].start, 200)
        assert np.isclose(f.markers["test_marker2"].stop, 300)


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
    assert f.point_scans == {}
    assert f.fdcurves == {}


def test_groups(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 1:
        assert str(f["Force HF"]) == "{'Force 1x', 'Force 1y'}"

        for x in range(0, 2):
            t = [name for name in f]
            assert set(t) == set(f)

        for x in range(0, 2):
            t = [name for name in f["Force HF"]]
            assert set(t) == set(["Force 1x", "Force 1y"])


def test_redirect_list(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        with pytest.warns(FutureWarning):
            f["Calibration"]

        with pytest.warns(FutureWarning):
            f["Marker"]

        with pytest.warns(FutureWarning):
            f["FD Curve"]

        with pytest.warns(FutureWarning):
            f["Kymograph"]

        with pytest.warns(FutureWarning):
            f["Scan"]


def test_repr_and_str(h5_file):
    f = pylake.File.from_h5py(h5_file)

    assert repr(f) == f"lumicks.pylake.File('{h5_file.filename}')"
    if f.format_version == 1:
        assert str(f) == dedent("""\
            File root metadata:
            - Bluelake version: unknown
            - Description: 
            - Experiment: 
            - Export time (ns): -1
            - File format version: 1
            - GUID: 
        
            Force HF:
              Force 1x:
              - Data type: float64
              - Size: 5
              Force 1y:
              - Data type: float64
              - Size: 5
            Force LF:
              Force 1x:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2
              Force 1y:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2
            
            .force1x
            .force1y
        """)
    if f.format_version == 2:
        assert str(f) == dedent("""\
            File root metadata:
            - Bluelake version: unknown
            - Description: 
            - Experiment: 
            - Export time (ns): -1
            - File format version: 2
            - GUID: 
            
            Force HF:
              Force 1x:
              - Data type: float64
              - Size: 5
              Force 1y:
              - Data type: float64
              - Size: 5
            Force LF:
              Force 1x:
              - Data type: [('Timestamp', '<i8'), ('Value', '<f8')]
              - Size: 2
              Force 1y:
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
              - test_marker
              - test_marker2

            .kymos
              - Kymo1
            
            .scans
              - fast X slow Z multiframe
              - fast Y slow X
              - fast Y slow X multiframe
              - fast Y slow Z multiframe
            
            .force1x
              .calibration
            .force1y
              .calibration
        """)


def test_invalid_file_format(h5_file_invalid_version):
    with pytest.raises(Exception):
        f = pylake.File.from_h5py(h5_file_invalid_version)


def test_invalid_access(h5_file):
    f = pylake.File.from_h5py(h5_file)

    if f.format_version == 2:
        with pytest.warns(FutureWarning):
            m = f["Kymograph"]

        with pytest.raises(IndexError):
            m["Kymo1"]
