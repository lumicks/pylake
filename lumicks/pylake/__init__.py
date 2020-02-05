from .__about__ import (__author__, __copyright__, __doc__, __email__, __license__, __summary__,
                        __title__, __url__, __version__)

from .file import *
from .correlated_stack import CorrelatedStack
from .fitting.fdmodels import force_model
from .fitting.fitobject import FitObject


def pytest(args=None, plugins=None):
    """Run the tests

    Parameters
    ----------
    args : list or str
        Command line options for pytest (excluding the target file/dir).
    plugins : list
        Plugin objects to be auto-registered during initialization.
    """
    import pytest
    import pathlib

    args = args or []
    module_path = str(pathlib.Path(__file__).parent)
    args.append(module_path)

    error_code = pytest.main(args, plugins)
    return error_code or None
