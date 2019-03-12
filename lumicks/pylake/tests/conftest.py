import h5py
import numpy as np
import pytest


# We generate mock data files for different versions of the Bluelake HDF5 file
# format:

class MockDataFile_v1:

    def __init__(self, file):
        self.file = h5py.File(file)

    def get_file_format_version(self):
        return 1

    def write_metadata(self):
        self.file.attrs["Bluelake version"] = "unknown"
        self.file.attrs["File format version"] = self.get_file_format_version()
        self.file.attrs["Experiment"] = ""
        self.file.attrs["Description"] = ""
        self.file.attrs["GUID"] = ""
        self.file.attrs["Export time (ns)"] = -1

    def make_continuous_channel(self, group, name, start, dt, data):
        if group not in self.file:
            self.file.create_group(group)

        self.file[group][name] = data
        dset = self.file[group][name]
        dset.attrs["Start time (ns)"] = start
        dset.attrs["Stop time (ns)"] = start + len(data) * dt
        dset.attrs["Sample rate (Hz)"] = 1 / dt * 1e9
        return dset

    def make_timeseries_channel(self, group, name, data):
        if group not in self.file:
            self.file.create_group(group)

        compound_type = np.dtype([("Timestamp", np.int64), ("Value", float)])
        self.file[group][name] = np.array(data, compound_type)
        dset = self.file[group][name]
        return dset

    def make_timetags_channel(self, group, name, data):
        raise NotImplementedError


class MockDataFile_v2(MockDataFile_v1):

    def get_file_format_version(self):
        return 2

    def make_continuous_channel(self, group, name, start, dt, data):
        dset = super().make_continuous_channel(group, name, start, dt, data)
        dset.attrs["Kind"] = "Continuous"

    def make_timeseries_channel(self, group, name, data):
        dset = super().make_timeseries_channel(group, name, data)
        dset.attrs["Kind"] = b"TimeSeries"

    def make_timetags_channel(self, group, name, data):
        if group not in self.file:
            self.file.create_group(group)

        self.file[group][name] = data
        dset = self.file[group][name]
        dset.attrs["Kind"] = "TimeTags"
        return dset


@pytest.fixture(scope="session", params=[MockDataFile_v1, MockDataFile_v2])
def h5_file(tmpdir_factory, request):
    mock_class = request.param

    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = mock_class(tmpdir.join("%s.h5" % mock_class.__class__.__name__))
    mock_file.write_metadata()

    mock_file.make_continuous_channel("Force HF", "Force 1x", 1, 10, np.arange(5.0))
    mock_file.make_continuous_channel("Force HF", "Force 1y", 1, 10, np.arange(5.0, 10.0))

    mock_file.make_timeseries_channel("Force LF", "Force 1x", [(1, 1.1), (2, 2.1)])
    mock_file.make_timeseries_channel("Force LF", "Force 1y", [(1, 1.2), (2, 2.2)])

    if mock_class == MockDataFile_v2:
        mock_file.make_timetags_channel(
            "Photon Time Tags", "Red",
            np.arange(10, 100, step=10, dtype=np.int64))

    return mock_file.file


@pytest.fixture(scope="session")
def h5_file_invalid_version(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("pylake")

    mock_file = h5py.File(tmpdir.join("invalid.h5"))
    mock_file.attrs["Bluelake version"] = "unknown"
    mock_file.attrs["File format version"] = 254

    return mock_file

