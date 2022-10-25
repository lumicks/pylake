import importlib
import warnings
import pytest
import matplotlib.pyplot as plt


def pytest_addoption(parser):
    for option in ("slow", "preflight"):
        parser.addoption(
            f"--run{option}", action="store_true", default=False, help=f"run {option} tests"
        )


def pytest_collection_modifyitems(config, items):
    nb_packages = ("ipywidgets", "notebook")
    has_notebook = all(importlib.util.find_spec(lib) for lib in nb_packages)
    if not has_notebook:
        skip_nb = pytest.mark.skip(reason=f"{nb_packages} need to be installed for these tests")
        for item in items:
            if "notebook" in item.keywords:
                item.add_marker(skip_nb)

    for option in ("slow", "preflight"):
        if config.getoption(f"--run{option}"):
            continue
        skip_slow = pytest.mark.skip(reason=f"need --run{option} option to run")
        for item in items:
            if option in item.keywords:
                item.add_marker(skip_slow)


def pytest_configure(config):
    # Use a headless backend for testing
    plt.switch_backend("agg")

    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "preflight: mark preflight tests which should only be run manually"
    )
    config.addinivalue_line(
        "markers", "notebook: these tests require the notebook dependencies to be installed"
    )


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
    # importing scipy submodules on some version of Python
    warnings.filterwarnings("ignore", category=ImportWarning)

    # bogus numpy ABI warning (see numpy/#432)
    warnings.filterwarnings(
        "ignore", category=ImportWarning, message=".*numpy.dtype size changed.*"
    )
    warnings.filterwarnings(
        "ignore", category=ImportWarning, message=".*numpy.ufunc size changed.*"
    )

    # Deprecation warnings from ipywidgets on import (types, _widget_types and _active_widgets)
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message=r".*Widget.(\w*) is deprecated.*"
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


@pytest.fixture(autouse=True)
def configure_mpl():
    try:
        with plt.style.context(["classic", "_classic_test_patch"]):
            yield
    finally:
        plt.close("all")
