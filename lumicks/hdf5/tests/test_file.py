import numpy as np
from lumicks import hdf5


def test_attributes(h5_file):
    f = hdf5.File.from_h5py(h5_file)

    assert type(f.bluelake_version) is str
    assert f.format_version == 1
    assert type(f.experiment) is str
    assert type(f.description) is str
    assert type(f.guid) is str
    assert np.issubdtype(f.export_time, int)


def test_channels(h5_file):
    f = hdf5.File.from_h5py(h5_file)

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
