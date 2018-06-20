import h5py
import numpy as np
import pytest


@pytest.fixture(scope="session")
def h5_file(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = h5py.File(tmpdir.join("tmp.h5"), 'w')

    mock_file.attrs["Bluelake version"] = "unknown"
    mock_file.attrs["File format version"] = 1
    mock_file.attrs["Experiment"] = ""
    mock_file.attrs["Description"] = ""
    mock_file.attrs["GUID"] = ""
    mock_file.attrs["Export time (ns)"] = -1

    def make_continuous_channel(group, name, start, dt, data):
        if group not in mock_file:
            mock_file.create_group(group)

        mock_file[group][name] = data
        dset = mock_file[group][name]
        dset.attrs["Start time (ns)"] = start
        dset.attrs["Stop time (ns)"] = start + len(data) * dt
        dset.attrs["Sample rate (Hz)"] = 1 / dt * 1e9

    make_continuous_channel("Force HF", "Force 1x", 1, 10, np.arange(5.0))
    make_continuous_channel("Force HF", "Force 1y", 1, 10, np.arange(5.0, 10.0))

    def make_timeseries_channel(group, name, data):
        if group not in mock_file:
            mock_file.create_group(group)

        compound_type = np.dtype([("Timestamp", np.int64), ("Value", float)])
        mock_file[group][name] = np.array(data, compound_type)

    make_timeseries_channel("Force LF", "Force 1x", [(1, 1.1), (2, 2.1)])
    make_timeseries_channel("Force LF", "Force 1y", [(1, 1.2), (2, 2.2)])

    return mock_file
