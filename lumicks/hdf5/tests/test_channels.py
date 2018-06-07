import pytest
import numpy as np
from lumicks.hdf5 import channel


def test_slice_properties():
    size = 5
    s = channel.Slice(np.random.rand(size), np.random.rand(size))
    assert len(s) == size

    s = channel.Slice([], [])
    assert len(s) == 0


def test_indexing():
    """The default integer indices are in timestamps (ns)"""
    s = channel.Slice([14, 15, 16, 17], [4, 5, 6, 7])

    # Scalar access
    assert s[4].data == 14
    assert s[4].timestamps == 4
    assert len(s[99]) == 0
    assert len(s[99]) == 0

    # Slices
    np.testing.assert_equal(s[4:5].data, [14])
    np.testing.assert_equal(s[4:5].timestamps, [4])

    np.testing.assert_equal(s[4:6].data, [14, 15])
    np.testing.assert_equal(s[4:6].timestamps, [4, 5])

    np.testing.assert_equal(s[4:10].data, [14, 15, 16, 17])
    np.testing.assert_equal(s[4:10].timestamps, [4, 5, 6, 7])

    with pytest.raises(IndexError) as exc:
        assert s[1:2:3]
    assert str(exc.value) == "Slice steps are currently not supported"

    s = channel.Slice([], [])
    assert len(s[0].data) == 0
    assert len(s[0].timestamps) == 0
    assert len(s[1:2].data) == 0
    assert len(s[1:2].timestamps) == 0


def test_asarray():
    """Slices can be given to numpy functions"""
    s = channel.Slice([14, 15, 16, 17], [4, 5, 6, 7])

    np.testing.assert_equal(np.asarray(s), s.data)
    assert np.sum(s) == np.sum(s.data)


def test_inspections(h5_file):
    assert channel.is_continuous_channel(h5_file["Force HF"]["Force 1x"]) is True
    assert channel.is_continuous_channel(h5_file["Force LF"]["Force 1x"]) is False


def test_channel(h5_file):
    force = channel.make_continuous_channel(h5_file["Force HF"]["Force 1x"])
    assert np.allclose(force.data, [0, 1, 2, 3, 4])
    assert np.allclose(force.timestamps, [1, 11, 21, 31, 41])

    downsampled = channel.make_timeseries_channel(h5_file["Force LF"]["Force 1x"])
    assert np.allclose(downsampled.data, [1.1, 2.1])
    assert np.allclose(downsampled.timestamps, [1, 2])
