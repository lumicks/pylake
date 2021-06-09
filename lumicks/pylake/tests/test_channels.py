import h5py
import pytest
import numpy as np
from lumicks.pylake import channel
from lumicks.pylake.calibration import ForceCalibration
import matplotlib as mpl
from matplotlib.testing.decorators import cleanup


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


def test_start_stop():
    s = channel.Slice(channel.TimeSeries([14, 15, 16, 17], [4, 6, 8, 10]))
    np.testing.assert_allclose(s.start, 4)
    np.testing.assert_allclose(s.stop, 10 + 1)

    s = channel.Slice(channel.Continuous([14, 15, 16, 17], 4, 2))
    np.testing.assert_allclose(s.start, 4)
    np.testing.assert_allclose(s.stop, 12)

    s = channel.Slice(channel.TimeTags([14, 15, 16, 17]))
    np.testing.assert_allclose(s.start, 14)
    np.testing.assert_allclose(s.stop, 17 + 1)

    s = channel.Slice(channel.TimeTags([14, 15, 16, 17], 4, 30))
    np.testing.assert_allclose(s.start, 4)
    np.testing.assert_allclose(s.stop, 30)


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
    np.testing.assert_allclose(force.data, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(force.timestamps, [1, 11, 21, 31, 41])

    downsampled = channel.TimeSeries.from_dataset(h5_file["Force LF"]["Force 1x"])
    np.testing.assert_allclose(downsampled.data, [1.1, 2.1])
    np.testing.assert_allclose(downsampled.timestamps, [1, 2])

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


def test_seconds_property():
    s = channel.Slice(channel.Continuous([14, 15, 16, 17], start=40, dt=1e9))
    np.testing.assert_allclose(s.seconds, [0, 1, 2, 3])

    s = channel.Slice(channel.TimeSeries([14, 15, 16, 17], [40e9, 41e9, 42e9, 43e9]))
    np.testing.assert_allclose(s.seconds, [0, 1, 2, 3])


def test_continuous_downsampling_to():
    # Continuous
    d = np.arange(1, 24)
    s = channel.Slice(channel.Continuous(d, 0, 500))  # 2 MHz

    # to 1000 ns step
    s2a = s.downsampled_to(1e6, where='left')
    assert np.all(np.equal(s2a.timestamps, [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]))
    np.testing.assert_allclose(s2a.data, [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5, 21.5])

    s2b = s.downsampled_to(1e6, where='center')
    assert np.all(np.equal(s2b.timestamps, (np.arange(0, 10500, 1000) + np.arange(500, 10501, 1000)) / 2))
    np.testing.assert_allclose(s2a.data, [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5, 21.5])

    # upsampling
    with pytest.raises(ValueError):
        s.downsampled_to(3e6, where="left")

    # non-integer ratio
    with pytest.raises(ValueError):
        s.downsampled_to(3e5, where="left")

    # non-integer ratio
    s3a = s.downsampled_to(3e5, where='left', method="ceil")
    assert np.all(np.equal(s3a.timestamps, [0, 3000, 6000]))
    np.testing.assert_allclose(s3a.data, [3.5, 9.5, 15.5])

    s3b = s.downsampled_to(3e5, where='center', method="ceil")
    assert np.all(np.equal(s3b.timestamps, [(0+2500)/2, (3000+5500)/2, (6000+8500)/2]))
    np.testing.assert_allclose(s3a.data, [3.5, 9.5, 15.5])


def test_continuous_like_downsampling_to():
    # timesteps = 500 ns
    # frequencies = 2 MHz
    t = np.arange(0, 11001, 500)
    d = np.arange(1, t.size + 1)
    s = channel.Slice(channel.TimeSeries(d, t))

    # to 1000 ns step
    s2a = s.downsampled_to(1e6, where='left')
    assert np.all(np.equal(s2a.timestamps, [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]))
    np.testing.assert_allclose(s2a.data, [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5, 21.5])

    s2b = s.downsampled_to(1e6, where='center')
    assert np.all(np.equal(s2b.timestamps, (np.arange(0, 10500, 1000) + np.arange(500, 10501, 1000)) / 2))
    np.testing.assert_allclose(s2a.data, [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5, 21.5])

    # upsampling
    with pytest.raises(ValueError):
        s.downsampled_to(3e6, where="left")

    # non-integer ratio
    with pytest.raises(ValueError):
        s.downsampled_to(3e5, where="left")

    # force non-integer ratio
    s3a = s.downsampled_to(3e5, where='left', method="ceil")
    assert np.all(np.equal(s3a.timestamps, [0, 3000, 6000]))
    np.testing.assert_allclose(s3a.data, [3.5, 9.5, 15.5])
    s3b = s.downsampled_to(3e5, where='center', method="ceil")
    assert np.all(np.equal(s3b.timestamps, [(0+2500)/2, (3000+5500)/2, (6000+8500)/2]))
    np.testing.assert_allclose(s3a.data, [3.5, 9.5, 15.5])


def test_variable_downsampling_to():
    # timesteps = 500, 10000 ns
    # frequencies = 2 MHz, 1 MHz
    t = np.hstack((np.arange(0, 5001, 500),
                   np.arange(6000, 14001, 1000)))
    d = np.arange(1, t.size + 1)
    s = channel.Slice(channel.TimeSeries(data=d, timestamps=t))

    with pytest.raises(ValueError):
        s.downsampled_to(3e6)

    with pytest.raises(ValueError):
        s.downsampled_to(5e5, where="left")

    # to 2000 ns step
    s2a = s.downsampled_to(5e5, where="left", method="force")
    assert np.all(np.equal(s2a.timestamps, [0, 2000, 4000, 6000, 8000, 10000, 12000]))
    np.testing.assert_allclose(s2a.data, [2.5, 6.5, 10.0, 12.5, 14.5, 16.5, 18.5])
    s2b = s.downsampled_to(5e5, where="center", method="force")

    # Original samples are 0 500 1000 1500 2000 2500 3000 3500 4000 4500 5000   6000 7000 8000 ...
    assert np.all(np.equal(s2b.timestamps, np.array(
                    [
                        (2000 - 500) / 2,
                        (2000 + 4000 - 500) / 2,
                        (4000 + 5000) / 2,
                        (6000 + 8000 - 1000) / 2,
                        (8000 + 10000 - 1000) / 2,
                        (10000 + 12000 - 1000) / 2,
                        (12000 + 14000 - 1000) / 2,
                    ]
                ).astype(np.int64)))
    np.testing.assert_allclose(s2b.data, [2.5, 6.5, 10.0, 12.5, 14.5, 16.5, 18.5])

    # to 3333 ns step
    s3a = s.downsampled_to(3e5, where="left", method="force")
    assert np.all(np.equal(s3a.timestamps, [0, 3333, 6666, 9999]))
    np.testing.assert_allclose(s3a.data, [4.0, 10.0, 14.0, 17.5])
    s3b = s.downsampled_to(3e5, where="center", method="force")

    assert np.all(np.equal(s3b.timestamps, [3000/2, (3000 + 6500)/2, (7000 + 9000)/2, (10000 + 13000)/2]))
    np.testing.assert_allclose(s3b.data, [4.0, 10.0, 14.0, 17.5])

    with pytest.raises(ValueError):
        s.downsampled_to(3e8)


def test_downsampling_consistency():
    d = np.arange(1, 24)
    s = channel.Slice(channel.Continuous(d, 0, 10))

    # Multiple of 5 should downsample to the same irrespective of the method
    # Source frequency was 1e9 / 10 Hz. So we go to .2e8 Hz.
    s1 = s.downsampled_to(.2e8)
    s2 = s.downsampled_by(5)
    np.testing.assert_allclose(s1.data, s2.data)
    np.testing.assert_allclose(s1.timestamps, s2.timestamps)

    d = np.arange(1, 24)
    s = channel.Slice(channel.Continuous(d, 5, 10))

    # Multiple of 5 should downsample to the same irrespective of the method
    # Source frequency was 1e9 / 10 Hz. So we go to .2e8 Hz.
    s1 = s.downsampled_to(.2e8)
    s2 = s.downsampled_by(5)
    np.testing.assert_allclose(s1.data, s2.data)
    np.testing.assert_allclose(s1.timestamps, s2.timestamps)

    # Multiple of 5 should downsample to the same irrespective of the method
    # Source frequency was 1e9 / 7 Hz.
    d = np.arange(1, 24)
    s = channel.Slice(channel.Continuous(d, 0, 7))
    s1 = s.downsampled_to(int(1e9 / 7 / 5))
    s2 = s.downsampled_by(5)
    np.testing.assert_allclose(s1.data, s2.data)
    np.testing.assert_allclose(s1.timestamps, s2.timestamps)


def test_consistency_downsampled_to():
    d = np.arange(1, 41)
    s = channel.Slice(channel.Continuous(d, 50, 10))

    one_step = s.downsampled_to(.1e8)
    two_step = s.downsampled_to(.2e8).downsampled_to(.1e8)

    np.testing.assert_allclose(one_step.data, two_step.data)
    np.testing.assert_allclose(one_step.timestamps, two_step.timestamps)


def test_downsampled_over_no_data_gap():
    t = np.array([0, 1, 2, 3, 10, 11, 12, 13, 14, 15])
    d = np.arange(10)
    s = channel.Slice(channel.TimeSeries(d, t))
    ranges = [(t1, t2) for t1, t2 in zip(np.arange(0, 16, 2), np.arange(2, 18, 2))]
    ts = s.downsampled_over(ranges)
    np.testing.assert_allclose(ts.timestamps, [0, 2, 10, 12, 14])
    np.testing.assert_allclose(ts.data, [0.5, 2.5, 4.5, 6.5, 8.5])


def test_downsampling_over_subset():
    d = np.arange(1, 24)
    s = channel.Slice(channel.Continuous(d, 0, 10))

    sd = s.downsampled_over([(20, 40), (40, 60), (60, 80)])
    # Data starts at 1, timestamps start at 0. 20-40 corresponds to data [3, 4], 40-60 to [5,6] etc.
    np.testing.assert_allclose(sd.data, [(3+4)/2, (5+6)/2, (7+8)/2])
    np.testing.assert_allclose(sd.timestamps, [(20+30)/2, (40+50)/2, (60+70)/2])


def test_downsampling_like():
    d = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
    s = channel.Slice(channel.Continuous(d, 100, 2))

    t_downsampled = np.array([0, 4, 8, 12, 16, 34, 40, 46, 50, 54]) + 100
    y_downsampled = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
    reference = channel.Slice(channel.TimeSeries(y_downsampled, t_downsampled))

    ds, ref_out = s.downsampled_like(reference)
    assert np.all(np.equal(ds.timestamps, ref_out.timestamps))
    np.testing.assert_allclose(t_downsampled[1:-1], ds.timestamps)
    np.testing.assert_allclose(y_downsampled[1:-1], ds.data)

    with pytest.raises(NotImplementedError):
        reference.downsampled_like(reference)

    with pytest.raises(AssertionError):
        s.downsampled_like(s)

@cleanup
def test_channel_plot():
    def testLine(x, y):
        data = [obj for obj in mpl.pyplot.gca().get_children() if isinstance(obj, mpl.lines.Line2D)]
        assert len(data) == 1
        line = data[0].get_data()
        np.testing.assert_allclose(line[0], x)
        np.testing.assert_allclose(line[1], y)

    d = np.arange(1, 24)
    s = channel.Slice(channel.Continuous(d, int(5e9), int(10e9)))
    s.plot()
    testLine(np.arange(0, 230, 10), d)

    mpl.pyplot.gca().clear()
    s.plot(start=0)
    testLine(np.arange(5, 230, 10), d)

    mpl.pyplot.gca().clear()
    s.plot(start=100e9)
    testLine(np.arange(0, 230, 10) - 100 + 5, d)


def test_regression_lazy_loading(h5_file):
    ch = channel.Continuous.from_dataset(h5_file["Force HF"]["Force 1x"])
    assert type(ch._src._src_data) == h5py.Dataset


@pytest.mark.parametrize(
    "data, new_data",
    [
        (channel.Continuous([1, 2, 3, 4, 5], start=1, dt=1), np.array([5, 6, 7, 8, 9])),
        (channel.TimeSeries([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]), np.array([5, 6, 7, 8, 9])),
    ],
)
def test_with_data(data, new_data):
    np.testing.assert_allclose(data._with_data(new_data).data, new_data)
    old_timestamps = data.timestamps
    np.testing.assert_allclose(data._with_data(new_data).timestamps, old_timestamps)
