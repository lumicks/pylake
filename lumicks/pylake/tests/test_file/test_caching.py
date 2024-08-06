import pytest

from lumicks import pylake


def test_global_cache(h5_file):
    # Load the file (never storing the file handle)
    f1x1 = pylake.File.from_h5py(h5_file).force1x
    f1x2 = pylake.File.from_h5py(h5_file).force1x

    # These should point to the same data
    assert id(f1x1.data) == id(f1x2.data)

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        f1x1.data[5:100] = 3

    file = pylake.File.from_h5py(h5_file)
    assert id(file.force1x.data) == id(file.force1x.data)
