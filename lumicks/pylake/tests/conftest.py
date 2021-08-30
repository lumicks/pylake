import h5py
import numpy as np
import pytest
import json
import warnings
import matplotlib.pyplot as plt
from .data.mock_confocal import generate_scan_json
from .data.mock_fdcurve import generate_fdcurve_with_baseline_offset


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_configure(config):
    # Use a headless backend for testing
    plt.switch_backend('agg')
    config.addinivalue_line("markers", "slow: mark test as slow to run")


# We generate mock data files for different versions of the Bluelake HDF5 file
# format:

class MockDataFile_v1:

    def __init__(self, file):
        self.file = h5py.File(file, 'w')

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

    def make_calibration_data(self, calibration_idx, group, attributes):
        if "Calibration" not in self.file:
            self.file.create_group("Calibration")

        # Numeric value converted to string
        if calibration_idx not in self.file["Calibration"]:
            self.file["Calibration"].create_group(calibration_idx)

        # e.g. Force 1x, Force 1y ... etc
        if group not in self.file["Calibration"][calibration_idx]:
            self.file["Calibration"][calibration_idx].create_group(group)

        # Attributes
        field = self.file["Calibration"][calibration_idx][group]
        for i, v in attributes.items():
            field.attrs[i] = v

    def make_fd(self, fd_name=None, metadata={}, attributes={}):
        if "FD Curve" not in self.file:
            self.file.create_group("FD Curve")

        if fd_name:
            dset = self.file["FD Curve"].create_dataset(fd_name, data=metadata)
            for i, v in attributes.items():
                dset.attrs[i] = v

    def make_marker(self, marker_name, attributes):
        if "Marker" not in self.file:
            self.file.create_group("Marker")

        if marker_name not in self.file["Marker"]:
            dset = self.file["Marker"].create_dataset(marker_name, data=f'{{"name":"{marker_name}"}}')

            for i, v in attributes.items():
                dset.attrs[i] = v

    def make_continuous_channel(self, group, name, start, dt, data):
        dset = super().make_continuous_channel(group, name, start, dt, data)
        dset.attrs["Kind"] = "Continuous"

    def make_timeseries_channel(self, group, name, data):
        dset = super().make_timeseries_channel(group, name, data)
        dset.attrs["Kind"] = b"TimeSeries"

    def make_json_data(self, group, name, data):
        if group not in self.file:
            self.file.create_group(group)

        self.file[group].create_dataset(name, data=data)
        return self.file[group][name]

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
    mock_file.make_continuous_channel("Force HF", "Force 1z", 1, 10, np.arange(10.0, 15.0))

    mock_file.make_timeseries_channel("Force LF", "Force 1x", [(1, 1.1), (2, 2.1)])
    mock_file.make_timeseries_channel("Force LF", "Force 1y", [(1, 1.2), (2, 2.2)])
    mock_file.make_timeseries_channel("Force LF", "Force 1z", [(1, 1.3), (2, 2.3)])

    if mock_class == MockDataFile_v2:
        mock_file.make_timetags_channel(
            "Photon Time Tags", "Red",
            np.arange(10, 100, step=10, dtype=np.int64))

        calibration_time_field = "Stop time (ns)"
        mock_file.make_calibration_data("1", "Force 1x", {calibration_time_field: 0})
        mock_file.make_calibration_data("2", "Force 1x", {calibration_time_field: 1})
        mock_file.make_calibration_data("3", "Force 1x", {calibration_time_field: 10})
        mock_file.make_calibration_data("4", "Force 1x", {calibration_time_field: 100})
        mock_file.make_calibration_data("1", "Force 1y", {calibration_time_field: 0})
        mock_file.make_calibration_data("2", "Force 1y", {calibration_time_field: 1})
        mock_file.make_calibration_data("3", "Force 1y", {calibration_time_field: 10})
        mock_file.make_calibration_data("4", "Force 1y", {calibration_time_field: 100})
        mock_file.make_calibration_data("1", "Force 1z", {calibration_time_field: 0})
        mock_file.make_calibration_data("2", "Force 1z", {calibration_time_field: 1})
        mock_file.make_calibration_data("3", "Force 1z", {calibration_time_field: 10})
        mock_file.make_calibration_data("4", "Force 1z", {calibration_time_field: 100})

        mock_file.make_marker("test_marker", {'Start time (ns)': 100, 'Stop time (ns)': 200})
        mock_file.make_marker("test_marker2", {'Start time (ns)': 200, 'Stop time (ns)': 300})
        mock_file.make_fd()

        counts = np.uint32([2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 8, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0,
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0,
                            0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 8, 0])

        infowave = np.uint8([1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 2,
                             0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2,
                             1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 2,
                             0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2])

        json_kymo = generate_scan_json([{"axis": 0, "num of pixels": 5, "pixel size (nm)": 10.0}])

        # Generate lines at 1 Hz
        freq = 1e9 / 16
        mock_file.make_continuous_channel("Photon count", "Red", np.int64(20e9), freq, counts)
        mock_file.make_continuous_channel("Photon count", "Green", np.int64(20e9), freq, counts)
        mock_file.make_continuous_channel("Photon count", "Blue", np.int64(20e9), freq, counts)
        mock_file.make_continuous_channel("Info wave", "Info wave", np.int64(20e9), freq, infowave)

        ds = mock_file.make_json_data("Kymograph", "Kymo1", json_kymo)
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)
        # Force channel that overlaps kymo; step from high to low force
        # We want two lines of the kymo to have a force of 30, the other 10. Force starts 5 samples
        # before the kymograph. First kymoline is 15 samples long, second is 16 samples long, which
        # means the third line starts after 31 + 5 = 36 samples
        force_data = np.hstack((np.ones(37) * 30,
                                np.ones(33) * 10))
        force_start = np.int64(ds.attrs["Start time (ns)"] - (freq*5))  # before infowave
        mock_file.make_continuous_channel("Force HF", "Force 2x", force_start, freq, force_data)
        mock_file.make_calibration_data("1", "Force 2x", {calibration_time_field: 0})
        mock_file.make_calibration_data("2", "Force 2x", {calibration_time_field: 1})
        mock_file.make_calibration_data("3", "Force 2x", {calibration_time_field: 10})
        mock_file.make_calibration_data("4", "Force 2x", {calibration_time_field: 100})

        # Single frame image
        json = generate_scan_json(
            [
                {"axis": 1, "num of pixels": 4, "pixel size (nm)": 191.0},
                {"axis": 0, "num of pixels": 5, "pixel size (nm)": 197.0},
            ],
        )
        ds = mock_file.make_json_data("Scan", "fast Y slow X", json)
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

        # Multi frame image
        json = generate_scan_json(
            [
                {"axis": 1, "num of pixels": 4, "pixel size (nm)": 191.0},
                {"axis": 0, "num of pixels": 3, "pixel size (nm)": 197.0},
            ],
        )
        ds = mock_file.make_json_data("Scan", "fast Y slow X multiframe", json)
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

        # Multiframe frame image
        json = generate_scan_json(
            [
                {"axis": 0, "num of pixels": 4, "pixel size (nm)": 191.0},
                {"axis": 2, "num of pixels": 3, "pixel size (nm)": 197.0},
            ],
        )
        ds = mock_file.make_json_data("Scan", "fast X slow Z multiframe", json)
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

        # Multiframe frame image
        json = generate_scan_json(
            [
                {"axis": 1, "num of pixels": 4, "pixel size (nm)": 191.0},
                {"axis": 2, "num of pixels": 3, "pixel size (nm)": 197.0},
            ],
        )
        ds = mock_file.make_json_data("Scan", "fast Y slow Z multiframe", json)
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

        # Point Scan
        ps_json_string = generate_scan_json([])
        ds = mock_file.make_json_data("Point Scan", "PointScan1", ps_json_string)
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

    return mock_file.file


@pytest.fixture(scope="session")
def fd_h5_file(tmpdir_factory, request):
    mock_class = MockDataFile_v2
    tmpdir = tmpdir_factory.mktemp("fdcurves")
    mock_file = mock_class(tmpdir.join(f"{mock_class.__name__}.h5"))
    mock_file.write_metadata()

    p, data = generate_fdcurve_with_baseline_offset()

    # write data
    fd_metadata = {"Polynomial Coefficients": {f"Corrected Force {n+1}x": p for n in range(2)}}
    fd_attrs = {"Start time (ns)": data["LF"]["time"][0],
                "Stop time (ns)": data["LF"]["time"][-1]+1}
    mock_file.make_fd("fd1", metadata=json.dumps(fd_metadata), attributes=fd_attrs)

    obs_force_lf_data = [datum for datum in zip(data["LF"]["time"], data["LF"]["obs_force"])]
    distance_lf_data = [datum for datum in zip(data["LF"]["time"], data["LF"]["distance"])]
    hf_start_time = data["HF"]["time"][0]
    for n in (1, 2):
        for component in ("x", "y"):
            mock_file.make_timeseries_channel("Force LF", f"Force {n}{component}", obs_force_lf_data)
            mock_file.make_continuous_channel("Force HF", f"Force {n}{component}", hf_start_time, 3, data["HF"]["obs_force"])
        mock_file.make_continuous_channel("Force HF", f"Corrected Force {n}x", hf_start_time, 3, data["HF"]["true_force"])
        mock_file.make_timeseries_channel("Distance", f"Distance {n}", distance_lf_data)

    return mock_file.file, (data["LF"]["time"], data["LF"]["true_force"])


@pytest.fixture(scope="session", params=[MockDataFile_v1, MockDataFile_v2])
def h5_file_missing_meta(tmpdir_factory, request):
    mock_class = request.param

    tmpdir = tmpdir_factory.mktemp("pylake")
    mock_file = mock_class(tmpdir.join("%s.h5" % mock_class.__class__.__name__))
    mock_file.write_metadata()

    if mock_class == MockDataFile_v2:
        infowave = np.uint8([1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 2,
                             0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2,
                             1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 2,
                             0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2])

        enc = json.JSONEncoder()

        # Generate lines at 1 Hz
        freq = 1e9 / 16

        # Single frame image - NO metadata
        ds = mock_file.make_json_data("Scan", "fast Y slow X no meta", enc.encode({}))
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

        # Single frame image - ok loading
        ds = mock_file.make_json_data(
            "Scan",
            "fast Y slow X",
            generate_scan_json(
                [
                    {"axis": 1, "num of pixels": 4, "pixel size (nm)": 191.0},
                    {"axis": 0, "num of pixels": 5, "pixel size (nm)": 197.0}
                ]
            ),
        )
        ds.attrs["Start time (ns)"] = np.int64(20e9)
        ds.attrs["Stop time (ns)"] = np.int64(20e9 + len(infowave) * freq)

    return mock_file.file


@pytest.fixture(scope="session")
def h5_file_invalid_version(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("pylake")

    mock_file = h5py.File(tmpdir.join("invalid.h5"), 'w')
    mock_file.attrs["Bluelake version"] = "unknown"
    mock_file.attrs["File format version"] = 254

    return mock_file


@pytest.fixture(scope="session")
def report_line():
    import atexit

    def reporter(text):
        """Print this line to a report at the end of the testing procedure"""
        def report():
            print(text)

        atexit.register(report)

    return reporter


@pytest.fixture(autouse=True)
def configure_warnings():
    # make warnings into errors but ignore certain third-party extension issues
    warnings.filterwarnings("error")

    # importing scipy submodules on some version of Python
    warnings.filterwarnings("ignore", category=ImportWarning)

    # bogus numpy ABI warning (see numpy/#432)
    warnings.filterwarnings(
        "ignore", category=ImportWarning, message=".*numpy.dtype size changed.*"
    )
    warnings.filterwarnings(
        "ignore", category=ImportWarning, message=".*numpy.ufunc size changed.*"
    )

    # h5py triggers a numpy DeprecationWarning when accessing empty datasets (such as our json
    # fields). Here they pass a None shape argument where () is expected by numpy. This will likely
    # be fixed in next h5py release, see the following PR on h5py:
    #   https://github.com/h5py/h5py/pull/1780/files
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*None into shape arguments as an alias for \\(\\) is.*",
    )
