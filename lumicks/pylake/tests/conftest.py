import pytest
import json
import warnings
import matplotlib.pyplot as plt
from .data.mock_file import MockDataFile_v2
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
