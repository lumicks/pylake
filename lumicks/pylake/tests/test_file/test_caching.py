import pytest

from lumicks import pylake
from lumicks.pylake.detail.caching import _get_array


def test_global_cache_continuous(h5_file):
    pylake.set_cache_enabled(True)
    _get_array.cache_clear()

    # Load the file (never storing the file handle)
    f1x1 = pylake.File.from_h5py(h5_file)["Force HF/Force 1x"]
    f1x2 = pylake.File.from_h5py(h5_file).force1x
    assert _get_array.cache_info().hits == 0  # No cache used yet (lazy loading)

    # These should point to the same data
    assert id(f1x1.data) == id(f1x2.data)
    assert _get_array.cache_info().hits == 1
    assert _get_array.cache_info().currsize == 40

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        f1x1.data[5:100] = 3

    file = pylake.File.from_h5py(h5_file)
    assert id(file.force1x.data) == id(file.force1x.data)


def test_global_cache_timeseries(h5_file):
    pylake.set_cache_enabled(True)
    _get_array.cache_clear()

    f1x1 = pylake.File.from_h5py(h5_file).downsampled_force1x
    f1x2 = pylake.File.from_h5py(h5_file).downsampled_force1x
    assert _get_array.cache_info().hits == 0  # No cache used yet (lazy loading)

    # These should point to the same data
    assert id(f1x1.data) == id(f1x2.data)
    assert _get_array.cache_info().hits == 1
    assert _get_array.cache_info().currsize == 16
    assert id(f1x1.timestamps) == id(f1x2.timestamps)
    assert _get_array.cache_info().hits == 2
    assert _get_array.cache_info().currsize == 32

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        f1x1.data[5:100] = 3

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        f1x1.timestamps[5:100] = 3


def test_global_cache_timetags(h5_file):
    pylake.set_cache_enabled(True)
    if pylake.File.from_h5py(h5_file).format_version == 2:
        _get_array.cache_clear()
        tags1 = pylake.File.from_h5py(h5_file)["Photon Time Tags"]["Red"]
        tags2 = pylake.File.from_h5py(h5_file)["Photon Time Tags"]["Red"]
        assert _get_array.cache_info().hits == 0

        # These should point to the same data
        assert id(tags1.data) == id(tags2.data)
        assert _get_array.cache_info().hits == 1
        assert _get_array.cache_info().currsize == 72

        with pytest.raises(ValueError, match="assignment destination is read-only"):
            tags1.data[5:100] = 3
