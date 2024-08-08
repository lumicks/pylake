import pytest

from lumicks import pylake


def test_global_cache_continuous(h5_file):
    # Load the file (never storing the file handle)
    f1x1 = pylake.File.from_h5py(h5_file).force1x
    f1x2 = pylake.File.from_h5py(h5_file).force1x

    # These should point to the same data
    assert id(f1x1.data) == id(f1x2.data)

    # Load the file (never storing the file handle)
    f1x1 = pylake.File.from_h5py(h5_file)["Force HF/Force 1x"]
    f1x2 = pylake.File.from_h5py(h5_file).force1x

    # These should point to the same data
    assert id(f1x1.data) == id(f1x2.data)

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        f1x1.data[5:100] = 3

    file = pylake.File.from_h5py(h5_file)
    assert id(file.force1x.data) == id(file.force1x.data)


def test_global_cache_timeseries(h5_file):
    f1x1 = pylake.File.from_h5py(h5_file).downsampled_force1x
    f1x2 = pylake.File.from_h5py(h5_file).downsampled_force1x

    # These should point to the same data
    assert id(f1x1.data) == id(f1x2.data)
    assert id(f1x1.timestamps) == id(f1x2.timestamps)

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        f1x1.data[5:100] = 3

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        f1x1.timestamps[5:100] = 3


def test_global_cache_timetags(h5_file):
    if pylake.File.from_h5py(h5_file).format_version == 2:
        tags1 = pylake.File.from_h5py(h5_file)["Photon Time Tags"]["Red"]
        tags2 = pylake.File.from_h5py(h5_file)["Photon Time Tags"]["Red"]

        # These should point to the same data
        assert id(tags1.data) == id(tags2.data)

        with pytest.raises(ValueError, match="assignment destination is read-only"):
            tags1.data[5:100] = 3
