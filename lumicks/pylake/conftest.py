import warnings
import importlib

import numpy as np
import pytest
import matplotlib.pyplot as plt


def pytest_addoption(parser):
    for option in ("slow", "preflight"):
        parser.addoption(
            f"--run{option}", action="store_true", default=False, help=f"run {option} tests"
        )
    parser.addoption(
        "--update_reference_data", action="store_true", default=False, help="Update test data"
    )
    parser.addoption(
        "--strict_reference_data",
        action="store_true",
        default=False,
        help="Treat missing test data as error",
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
            if any(item.iter_markers(name=option)):
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


@pytest.fixture
def reference_data(request):
    """Read or update test reference data.

    Some tests require relatively big reference matrices. If the purpose of the test is just to
    ensure that changes to the results are noticed, then this fixture can be used to store
    reference data. Simply import the fixture in the test, and call `reference_data` with a unique
    name per result and the dataset that needs to be stored. Check the resulting files into git to
    pin the results.

    If the functionality is ever updated, either delete the reference file and run the test, or run
    pytest with the option `--update_reference_data`. In order to explicitly disallow creation (and
    raise if the file is not found, run with the option `--strict_reference_data`).

    Examples
    --------
    ::

        @pytest.mark.parametrize("par", [1])
        def test_freezing(reference_data, par):
            test_data = np.array([[1, 2, 3], [1, 2, 3]])

            # Writes to a file function/freezing[parametrization descriptions].npz
            np.testing.assert_allclose(test_data, reference_data(test_data))

            # Writes to a file function/test_data[parametrization descriptions].npz
            np.testing.assert_allclose(test_data, reference_data(test_data, test_name="test_data"))

            # Writes to a file function/filename.npz. Note that when using this form, you are
            # responsible for assembling the parameterization into the `file_name`!
            np.testing.assert_allclose(test_data, reference_data(test_data, file_name="file_name"))
    """

    def get_reference_data(reference_data, test_name=None, file_name=None):
        """Read or write reference data

        Reads or writes reference data at the location `reference_data/test_name/dataset_name`.
        This function reads and returns reference data if it is available at the provided location.
        If it is not available, it writes a file containing the data (unless strict is enforced
        through `pytest` in which case it throws).

        `dataset_name` depends on the input arguments. By default, `dataset_name` will evaluate to
        the name of the test from which this function is called. The arguments `test_name` and
        `file_name` can override this behavior (see below).

        Parameters
        ----------
        reference_data : anything
            Data to store.
        test_name : str, optional
            If this is provided, the dataset name will be a combination of this string with the
            parametrization.
        file_name : str, optional
            If this is provided the dataset name will be exactly the file_name without the
            parametrization suffixed.

        Raises
        ------
        ValueError
            If both test_name and file_name are provided.
        """
        if test_name and file_name:
            raise ValueError("You cannot specify both a test_name and file_name")

        calling_function = request.function.__name__.replace("test_", "")

        # Force output file name.
        ref_data_filename = (
            request.node.name.replace("test_", "") if file_name is None else file_name
        )

        # Overwrite only the identifier part Test[par1-par2-par3] -> mytest[par1-par2,par3].
        if test_name is not None:
            ref_data_filename = ref_data_filename.replace(calling_function, test_name)

        calling_path = request.path.parent

        reference_data_path = calling_path / "reference_data" / calling_function
        reference_file_path = reference_data_path / f"{ref_data_filename}.npz"

        if reference_file_path.exists() and not request.config.getoption("--update_reference_data"):
            with np.load(reference_file_path, allow_pickle=True) as npz_file:
                return npz_file["arr_0"][()]
        else:
            if request.config.getoption("--strict_reference_data"):
                raise RuntimeError(
                    f"Test data for {calling_function}/{ref_data_filename} missing! Did you forget "
                    f"to check it in?"
                )

            reference_data_path.mkdir(parents=True, exist_ok=True)

            try:
                reference_data = np.asarray(reference_data)
            except ValueError:
                reference_data = np.asarray(reference_data, dtype=object)

            np.savez(reference_file_path, reference_data)
            print(f"\nWritten reference data {ref_data_filename} to {reference_file_path}.")
            return reference_data

    return get_reference_data
