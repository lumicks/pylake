import pytest
import numpy as np
from lumicks.pylake import channel
from lumicks.pylake.calibration import ForceCalibration


def test_calibration_timeseries_channels():
    time_field = 'Stop time (ns)'
    mock_calibration = ForceCalibration(time_field=time_field,
                                        items=[
                                           {'Calibration Data': 50, time_field: 50},
                                           {'Calibration Data': 20, time_field: 20},
                                           {'Calibration Data': 30, time_field: 30},
                                           {'Calibration Data': 40, time_field: 40},
                                           {'Calibration Data': 80, time_field: 80},
                                           {'Calibration Data': 90, time_field: 90},
                                           {'Calibration Data': 120, time_field: 120},
                                       ])

    # Channel should have calibration points 40, 50 since this is the only area that has force data.
    cc = channel.Slice(channel.TimeSeries([14, 15, 16, 17], [40, 50, 60, 70]),
                       calibration=mock_calibration)
    assert len(cc.calibration) == 2
    assert cc.calibration[0]["Calibration Data"] == 40
    assert cc.calibration[1]["Calibration Data"] == 50

    data = np.arange(14, 23, 1)
    time = np.arange(40, 130, 10)
    cc = channel.Slice(channel.TimeSeries(data, time), calibration=mock_calibration)
    assert len(cc.calibration) == 5

    calibration = cc[50:80].calibration
    assert len(calibration) == 1
    assert calibration[0]["Calibration Data"] == 50

    calibration = cc[50:90].calibration
    assert len(calibration) == 2
    assert calibration[0]["Calibration Data"] == 50
    assert calibration[1]["Calibration Data"] == 80

    # Check whether slice nesting works for calibration data
    # :120 => keeps 40, 50, 80, 90
    nested_slice = cc[:120]
    assert len(nested_slice.calibration) == 4
    assert nested_slice.calibration[0]["Calibration Data"] == 40
    assert nested_slice.calibration[1]["Calibration Data"] == 50
    assert nested_slice.calibration[2]["Calibration Data"] == 80
    assert nested_slice.calibration[3]["Calibration Data"] == 90
    nested_slice = nested_slice[:90]
    assert len(nested_slice.calibration) == 3

    # This slices off everything
    nested_slice = nested_slice[120:]
    assert len(nested_slice.calibration) == 0
    assert type(nested_slice.calibration) is list


def test_calibration_continuous_channels():
    time_field = 'Stop time (ns)'
    mock_calibration = ForceCalibration(time_field=time_field,
                                        items=[
                                            {'Calibration Data': 50, time_field: 50},
                                            {'Calibration Data': 20, time_field: 20},
                                            {'Calibration Data': 30, time_field: 30},
                                            {'Calibration Data': 40, time_field: 40},
                                            {'Calibration Data': 80, time_field: 80},
                                            {'Calibration Data': 90, time_field: 90},
                                            {'Calibration Data': 120, time_field: 120},
                                        ])

    # Channel should have calibration points 40, 50 since this is the only area that has force data.
    cc = channel.Slice(channel.Continuous([14, 15, 16, 17], 40, 10), calibration=mock_calibration)
    assert len(cc.calibration) == 2
    assert cc.calibration[0]["Calibration Data"] == 40
    assert cc.calibration[1]["Calibration Data"] == 50

    # Channel should have calibration points 40, 50, 80, 90, 120
    # and time points 40, 50, ... 120
    cc = channel.Slice(channel.Continuous(np.arange(14, 23, 1), 40, 10), calibration=mock_calibration)
    assert len(cc.calibration) == 5

    calibration = cc[50:80].calibration
    assert len(calibration) == 1
    assert calibration[0]["Calibration Data"] == 50

    calibration = cc[50:90].calibration
    assert len(calibration) == 2
    assert calibration[0]["Calibration Data"] == 50
    assert calibration[1]["Calibration Data"] == 80

    # Check whether slice nesting works for calibration data
    # :120 => keeps 40, 50, 80, 90
    nested_slice = cc[:120]
    assert len(nested_slice.calibration) == 4
    assert nested_slice.calibration[0]["Calibration Data"] == 40
    assert nested_slice.calibration[1]["Calibration Data"] == 50
    assert nested_slice.calibration[2]["Calibration Data"] == 80
    assert nested_slice.calibration[3]["Calibration Data"] == 90
    nested_slice = nested_slice[:90]
    assert len(nested_slice.calibration) == 3

    # 120 and up results in calibration point 120.
    # This case would be 80 if calibration data would be sliced every time, rather than filtered only when requested.
    nested_slice = nested_slice[120:]
    assert len(nested_slice.calibration) == 1
    assert nested_slice.calibration[0]["Calibration Data"] == 120


def test_slice_properties():
    size = 5
    s = channel.Slice(channel.TimeSeries(np.random.rand(size), np.random.rand(size)))
    assert len(s) == size
    assert s.sample_rate is None

    s = channel.Slice(channel.Continuous(np.random.rand(size), start=0, dt=1))
    assert len(s) == size
    assert s.sample_rate == 1e9

    size = 10
    s = channel.Slice(channel.TimeTags(np.arange(0, size, dtype=np.int64)))
    assert len(s) == size
    assert s.sample_rate is None

    s = channel.empty_slice
    assert len(s) == 0
    assert s.sample_rate is None


def test_labels():
    """Slicing must preserve labels"""
    size = 5
    labels = {"x": "distance", "y": "force"}
    s = channel.Slice(channel.TimeSeries(np.random.rand(size), np.random.rand(size)), labels)
    assert s.labels == labels
    assert s[:].labels == labels
    assert s[:0].labels == labels
    assert s[:10].labels == labels

    s = channel.Slice(channel.TimeSeries([], []), labels)
    assert len(s) == 0
    assert s.labels == labels
    assert s[:].labels == labels


def test_empty_slice():
    s = channel.empty_slice
    assert len(s[1:2].data) == 0
    assert len(s[1:2].timestamps) == 0


def test_timeseries_indexing():
    """The default integer indices are in timestamps (ns)"""
    s = channel.Slice(channel.TimeSeries([14, 15, 16, 17], [4, 5, 6, 7]))

    np.testing.assert_equal(s[0:5].data, [14])
    np.testing.assert_equal(s[0:5].timestamps, [4])
    np.testing.assert_equal(s[4:5].data, [14])
    np.testing.assert_equal(s[4:5].timestamps, [4])
    np.testing.assert_equal(s[4:6].data, [14, 15])
    np.testing.assert_equal(s[4:6].timestamps, [4, 5])
    np.testing.assert_equal(s[4:10].data, [14, 15, 16, 17])
    np.testing.assert_equal(s[4:10].timestamps, [4, 5, 6, 7])

    with pytest.raises(IndexError) as exc:
        assert s[1]
    assert str(exc.value) == "Scalar indexing is not supported, only slicing"
    with pytest.raises(IndexError) as exc:
        assert s[1:2:3]
    assert str(exc.value) == "Slice steps are not supported"

    s = channel.Slice(channel.TimeSeries([], []))
    assert len(s[1:2].data) == 0
    assert len(s[1:2].timestamps) == 0


def test_continuous_idexing():
    s = channel.Slice(channel.Continuous([14, 15, 16, 17], 4, 1))
    np.testing.assert_equal(s[0:5].data, [14])
    np.testing.assert_equal(s[0:5].timestamps, [4])
    np.testing.assert_equal(s[4:5].data, [14])
    np.testing.assert_equal(s[4:5].timestamps, [4])
    np.testing.assert_equal(s[4:6].data, [14, 15])
    np.testing.assert_equal(s[4:6].timestamps, [4, 5])
    np.testing.assert_equal(s[4:10].data, [14, 15, 16, 17])
    np.testing.assert_equal(s[4:10].timestamps, [4, 5, 6, 7])

    s = channel.Slice(channel.Continuous([14, 15, 16, 17], 4, 2))
    np.testing.assert_equal(s[0:5].data, [14])
    np.testing.assert_equal(s[0:5].timestamps, [4])
    np.testing.assert_equal(s[4:5].data, [14])
    np.testing.assert_equal(s[4:5].timestamps, [4])
    np.testing.assert_equal(s[4:8].data, [14, 15])
    np.testing.assert_equal(s[4:8].timestamps, [4, 6])
    np.testing.assert_equal(s[4:14].data, [14, 15, 16, 17])
    np.testing.assert_equal(s[4:14].timestamps, [4, 6, 8, 10])

    with pytest.raises(IndexError) as exc:
        assert s[1]
    assert str(exc.value) == "Scalar indexing is not supported, only slicing"
    with pytest.raises(IndexError) as exc:
        assert s[1:2:3]
    assert str(exc.value) == "Slice steps are not supported"

    s = channel.Slice(channel.TimeSeries([], []))
    assert len(s[1:2].data) == 0
    assert len(s[1:2].timestamps) == 0

    # Regression test for slicing within timestep
    s = channel.Slice(channel.Continuous([14, 15, 16, 17], 4, 2))
    assert s[5:15].timestamps[0] == 6
    s = channel.Slice(channel.Continuous([14, 15, 16, 17], -4, 2))
    assert s[-3:].timestamps[0] == -2
    s = channel.Slice(channel.Continuous([14, 15, 16, 17], 4, 3))
    assert s[3:15].timestamps[0] == 4
    s = channel.Slice(channel.Continuous([14, 15, 16, 17], 4, 3))
    assert s[4:15].timestamps[0] == 4
    assert s[5:15].timestamps[0] == 7
    assert s[6:15].timestamps[0] == 7
    assert s[7:15].timestamps[0] == 7
    assert s[8:15].timestamps[0] == 10
    assert s[6:14].timestamps[-1] == 13
    assert s[6:13].timestamps[-1] == 10


def test_timetags_indexing():
    s = channel.Slice(channel.TimeTags([10, 20, 30, 40, 50, 60]))
    np.testing.assert_equal(s[0:100].data, [10, 20, 30, 40, 50, 60])
    np.testing.assert_equal(s[10:100].data, [10, 20, 30, 40, 50, 60])
    np.testing.assert_equal(s[15:100].data, [20, 30, 40, 50, 60])
    np.testing.assert_equal(s[10:60].data, [10, 20, 30, 40, 50])
    np.testing.assert_equal(s[10:55].data, [10, 20, 30, 40, 50])
    np.testing.assert_equal(s[11:50].data, [20, 30, 40])
    np.testing.assert_equal(s[20:].data, [20, 30, 40, 50, 60])
    np.testing.assert_equal(s[:50].data, [10, 20, 30, 40])

    with pytest.raises(IndexError) as exc:
        assert s[1]
    assert str(exc.value) == "Scalar indexing is not supported, only slicing"
    with pytest.raises(IndexError) as exc:
        assert s[1:2:3]
    assert str(exc.value) == "Slice steps are not supported"

    s = channel.Slice(channel.TimeTags([]))
    assert len(s[10:30].data) == 0


def test_time_indexing():
    """String time-based indexing"""
    s = channel.Slice(channel.TimeSeries([1, 2, 3, 4, 5], [1400, 2500, 16e6, 34e9, 122 * 1e9]))
    # --> in time indices: ['0ns', '1100ns', '15.9986ms', '33.99s', '2m 2s']

    def assert_equal(actual, expected):
        np.testing.assert_equal(actual.data, expected)

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
        assert s['1ns']
    assert str(exc.value) == "Scalar indexing is not supported, only slicing"
    with pytest.raises(IndexError) as exc:
        assert s['1ns':'2s':'3ms']
    assert str(exc.value) == "Slice steps are not supported"

    s = channel.empty_slice
    assert len(s['1s':'2h'].data) == 0


def test_inspections(h5_file):
    assert channel.channel_class(h5_file["Force HF"]["Force 1x"]) == channel.Continuous
    assert channel.channel_class(h5_file["Force LF"]["Force 1x"]) == channel.TimeSeries
    if "Photon Time Tags" in h5_file:
        assert channel.channel_class(h5_file["Photon Time Tags"]["Red"]) == channel.TimeTags


def test_channel(h5_file):
    force = channel.Continuous.from_dataset(h5_file["Force HF"]["Force 1x"])
    assert np.allclose(force.data, [0, 1, 2, 3, 4])
    assert np.allclose(force.timestamps, [1, 11, 21, 31, 41])

    downsampled = channel.TimeSeries.from_dataset(h5_file["Force LF"]["Force 1x"])
    assert np.allclose(downsampled.data, [1.1, 2.1])
    assert np.allclose(downsampled.timestamps, [1, 2])

    if "Photon Time Tags" in h5_file:
        timetags = channel.TimeTags.from_dataset(h5_file["Photon Time Tags"]["Red"])
        assert np.all(np.equal(timetags.data, [10, 20, 30, 40, 50, 60, 70, 80, 90]))
        assert np.all(np.equal(timetags.timestamps, [10, 20, 30, 40, 50, 60, 70, 80, 90]))


def test_downsampling():
    s = channel.Slice(channel.Continuous([14, 15, 16, 17], start=40, dt=10))
    assert s.sample_rate == 1e8

    s2 = s.downsampled_by(2)
    np.testing.assert_allclose(s2.data, 14.5, 16.5)
    np.testing.assert_allclose(s2.timestamps, [45, 65])
    assert s2.sample_rate == 0.5e8

    s4 = s.downsampled_by(4)
    np.testing.assert_allclose(s4.data, 15.5)
    np.testing.assert_allclose(s4.timestamps, [55])
    assert s4.sample_rate == 0.25e8

    s3 = s.downsampled_by(3)
    np.testing.assert_allclose(s3.data, 15)
    np.testing.assert_allclose(s3.timestamps, [50])
    assert s3.sample_rate == 33333333

    s22 = s2.downsampled_by(2)
    np.testing.assert_allclose(s22.data, 15.5)
    np.testing.assert_allclose(s22.timestamps, [55])
    assert s22.sample_rate == 0.25e8

    with pytest.raises(ValueError):
        s.downsampled_by(-1)
    with pytest.raises(TypeError):
        s.downsampled_by(1.5)
