import pytest
import numpy as np
from lumicks.pylake import channel


def test_slice_properties():
    size = 5
    s = channel.Slice(np.random.rand(size), np.random.rand(size))
    assert len(s) == size

    s = channel.Slice([], [])
    assert len(s) == 0


def test_labels():
    """Slicing must preserve labels"""
    size = 5
    labels = {"x": "distance", "y": "force"}
    s = channel.Slice(np.random.rand(size), np.random.rand(size), labels)
    assert s.labels == labels
    assert s[:].labels == labels
    assert s[:0].labels == labels
    assert s[:10].labels == labels

    s = channel.Slice([], [], labels)
    assert len(s) == 0
    assert s.labels == labels
    assert s[:].labels == labels


def test_indexing():
    """The default integer indices are in timestamps (ns)"""
    s = channel.Slice([14, 15, 16, 17], [4, 5, 6, 7])

    # Scalar access
    assert s[4].data == 14
    assert s[4].timestamps == 4
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
    assert str(exc.value) == "Slice steps are not supported"

    s = channel.Slice([], [])
    assert len(s[0].data) == 0
    assert len(s[0].timestamps) == 0
    assert len(s[1:2].data) == 0
    assert len(s[1:2].timestamps) == 0


def test_time_indexing():
    """String time-based indexing"""
    s = channel.Slice([1, 2, 3, 4, 5], [1400, 2500, 16e6, 34e9, 122 * 1e9])
    # --> in time indices: ['0ns', '1100ns', '15.9986ms', '33.99s', '2m 2s']

    # Scalar access
    assert s[1400].data == 1
    assert s['0ns'].data == 1
    assert s['1100ns'].data == 2
    assert s['1.1us'].data == 2
    assert s['1us 100ns'].data == 2
    assert s['-1ns'].data == 5
    assert len(s['1ns']) == 0

    def assert_equal(actual, expected):
        np.testing.assert_equal(actual.data, expected)

    # Slices
    assert_equal(s['0ns':'1100ns'], [1])
    assert_equal(s['0ns':'1101ns'], [1, 2])
    assert_equal(s['1us':'1.1us'], [])
    assert_equal(s['1us':'1.2us'], [2])
    assert_equal(s['5ns':'17ms'], [2, 3])
    assert_equal(s['1ms':'40s'], [3, 4])
    assert_equal(s['0h':'2m 30s'], [1, 2, 3, 4, 5])
    assert_equal(s['0d':'2h'], [1, 2, 3, 4, 5])
    assert_equal(s['2m':'2.5m'], [5])
    assert_equal(s['2m':'2m 1s'], [])
    assert_equal(s['2m':'2m 3s'], [5])

    assert_equal(s[:'2.1s'], [1, 2, 3])
    assert_equal(s['2.1s':], [4, 5])
    assert_equal(s[:'-1s'], [1, 2, 3, 4])
    assert_equal(s[:'-2m'], [1, 2, 3])
    assert_equal(s[:'-5m'], [])
    assert_equal(s['-5m':], [1, 2, 3, 4, 5])
    assert_equal(s['-5m':], [1, 2, 3, 4, 5])

    with pytest.raises(IndexError) as exc:
        assert s['1ns':'2s':'3ms']
    assert str(exc.value) == "Slice steps are not supported"

    s = channel.Slice([], [])
    assert len(s['0ns'].data) == 0
    assert len(s['1s':'2h'].data) == 0


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
