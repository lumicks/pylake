import json
import hashlib
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

    # This is a warning that gets issued by IPython when calling pytest from a notebook. The backend
    # handling that used to be handled in IPython was moved to matplotlib. Until the mpl side of
    # this is out a warning is issued. See:
    #  - https://github.com/ipython/ipython/issues/14311
    #  - https://github.com/ipython/ipython/pull/14371
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*backend2gui is deprecated since IPython.*",
    )


@pytest.fixture(autouse=True)
def configure_mpl():
    try:
        with plt.style.context(["classic", "_classic_test_patch"]):
            yield
    finally:
        plt.close("all")


def filename_hash(string_to_hash):
    """Hash a filename for a test

    Parameters
    ----------
    string_to_hash : str
        String to return the hash of.

    We use a base36 hash to ensure that we can safely store these hashes as filenames on all
    operating systems. Base64 contains / and includes both small and capital case. This leads
    to issues on windows since its file systems are not case-sensitive. Base36 is still shorter
    than a hexadecimal hash.
    """
    md5_hash = hashlib.md5(string_to_hash.encode())
    return np.base_repr(int(md5_hash.hexdigest(), 16), 36)


def _json_read_write():
    def read_data(file_path):
        with open(file_path) as f:
            return json.load(f)["result"]

    def write_data(file_path, data, parametrization, calling_function):
        with open(file_path, "w") as f:
            json.dump(
                {
                    "test": calling_function,
                    "parametrization": parametrization,
                    "result": data,
                },
                f,
                indent=2,
            )
            f.write("\n")  # Text files should end with a newline

    return read_data, write_data, "json"


def _npz_read_write():
    def read_data(file_path):
        with np.load(file_path, allow_pickle=True) as npz_file:
            return npz_file["arr_0"][()]

    def write_data(file_path, data, _parametrization, _calling_function):
        try:
            data = np.asarray(data)
        except ValueError:
            data = np.asarray(data, dtype=object)

        np.savez(file_path, data)

    return read_data, write_data, "npz"


@pytest.fixture
def reference_data(request):
    """Read or update test reference data.

    Some tests require relatively big reference matrices. If the purpose of the test is just to
    ensure that changes to the results are noticed, then this fixture can be used to store
    reference data. Simply import the fixture in the test, and call `ref_data` with a unique
    name per result and the dataset that needs to be stored. Check the resulting files into git to
    pin the results.

    If the functionality is ever updated, either delete the reference file and run the test, or run
    pytest with the option `--update_reference_data`. In order to explicitly disallow creation (and
    raise if the file is not found, run with the option `--strict_reference_data`).

    Note that filenames with test parametrization are hashed to make sure we do not end up with
    excessively long filenames.

    Examples
    --------
    ::

        @pytest.mark.parametrize("par", [1])
        def test_freezing(ref_data, par):
            test_data = np.array([[1, 2, 3], [1, 2, 3]])

            # Writes to a file function/hash(freezing[parametrization descriptions]).npz
            np.testing.assert_allclose(test_data, ref_data(test_data))

            # Writes to a file function/hash(test_data[parametrization descriptions]).npz
            np.testing.assert_allclose(test_data, ref_data(test_data, test_name="test_data"))

            # Writes to a file function/filename.npz. Note that when using this form, you are
            # responsible for assembling the parametrization into the `file_name`!
            np.testing.assert_allclose(test_data, ref_data(test_data, file_name="file_name"))
    """

    def get_reference_data(reference_data, test_name=None, file_name=None, json=False):
        """Read or write reference data

        Reads or writes reference data at `ref_data/test_name/hash(dataset_name)`.
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
        json : bool
            Store as json rather than npz.

        Raises
        ------
        ValueError
            If both test_name and file_name are provided.
        """
        read_data, write_data, extension = _json_read_write() if json else _npz_read_write()

        if test_name and file_name:
            raise ValueError("You cannot specify both a test_name and file_name")

        calling_function = request.function.__name__.replace("test_", "")  # Just the name
        ref_data_filename = request.node.name.replace("test_", "")  # Name plus parametrization

        # Overwrite identifier part Test[par1-par2-par3] -> mytest[par1-par2,par3].
        if test_name is not None:
            ref_data_filename = ref_data_filename.replace(calling_function, test_name)

        # Fetch parametrization if it exists
        params = request.node.callspec.params if hasattr(request.node, "callspec") else {}

        # Parametrized test filenames can be overly long, hence we hash them.
        if params:
            ref_data_filename = filename_hash(ref_data_filename)

        # Override filename
        ref_data_filename = file_name if file_name else ref_data_filename

        calling_path = request.path.parent
        reference_data_path = calling_path / "ref_data"
        reference_file_path = reference_data_path / f"{ref_data_filename}.{extension}"

        if reference_file_path.exists() and not request.config.getoption("--update_reference_data"):
            return read_data(reference_file_path)
        else:
            if request.config.getoption("--strict_reference_data"):
                raise RuntimeError(
                    f"Test data for {calling_function} ({ref_data_filename}) missing! Did you "
                    f"forget to check the file {reference_file_path} in?"
                )

            reference_data_path.mkdir(parents=True, exist_ok=True)
            write_data(reference_file_path, reference_data, params, calling_function)

            print(f"\nWritten reference data {ref_data_filename} to {reference_file_path}.")
            return reference_data

    return get_reference_data


@pytest.fixture
def compare_to_reference_dict(reference_data):
    """Read or update test reference dictionary.

    Intended to store and/or compare to a reference dictionary stored in a human-readable json
    file. Only intended for single numeric values. See `ref_data` for usage information.

    Examples
    --------
    ::

        # Writes to a test file named freezing/freezing.json
        def test_freezing(compare_to_reference_dict):
            compare_to_reference_dict({"a": 5, "b": 1e-12}, rtol=1e-6)
    """

    def validate_dictionary_equality(data, test_name=None, file_name=None, rtol=1e-7, atol=0):
        def check_similarity(key, test_dict, ref_dict):
            if key not in test_dict:
                return f"{key}: missing vs {ref_dict[key]} (reference only)", False
            elif key not in ref_dict:
                return f"{key}: {test_dict[key]} vs missing (test only)", False
            elif not np.allclose(
                (test_value := test_dict[key]), (ref_value := ref_dict[key]), rtol=rtol, atol=atol
            ):
                return f"{key}: {test_value} vs {ref_value} (difference)", False
            else:
                return f"{key}: {test_value} vs {ref_value} (match)", True

        ref_data = reference_data(data, test_name=test_name, file_name=file_name, json=True)
        all_keys = (ref_data | data).keys()

        difference_str, match = zip(*[check_similarity(key, data, ref_data) for key in all_keys])

        if not all(match):
            raise RuntimeError(
                "Differences with reference data detected.\n" + "\n".join(difference_str)
            )

    return validate_dictionary_equality
