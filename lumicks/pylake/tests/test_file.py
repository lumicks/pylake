import numpy as np
from lumicks import pylake
import pytest
from textwrap import dedent


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


def test_properties(h5_file):
    f = pylake.File.from_h5py(h5_file)
    assert f.kymos == {}
    assert f.scans == {}
    assert f.point_scans == {}
    assert f.fdcurves == {}


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
        """)


def test_invalid_file_format(h5_file_invalid_version):
    with pytest.raises(Exception):
        f = pylake.File.from_h5py(h5_file_invalid_version)
