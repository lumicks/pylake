from .__about__ import (__author__, __copyright__, __doc__, __email__, __license__, __summary__,
                        __title__, __url__, __version__)

from .file import *
from .fitting.models import *
from .correlated_stack import CorrelatedStack
from .fitting.fit import FdFit
from lumicks.pylake.fitting.parameter_trace import parameter_trace
from lumicks.pylake.nb_widgets.range_selector import FdRangeSelector, FdDistanceRangeSelector
from lumicks.pylake.kymotracker.kymotracker import track_greedy, track_lines, filter_lines, refine_lines_centroid
from lumicks.pylake.nb_widgets.kymotracker_widgets import KymoWidgetGreedy
from lumicks.pylake.fdensemble import FdEnsemble


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
