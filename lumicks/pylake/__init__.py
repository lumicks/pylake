from .__about__ import (
    __author__,
    __copyright__,
    __doc__,
    __email__,
    __license__,
    __summary__,
    __title__,
    __url__,
    __version__,
)

from .file_download import *
from .benchmark import benchmark
from .adjustments import ColorAdjustment
from .correlated_stack import CorrelatedStack
from .piezo_tracking.piezo_tracking import *
from .piezo_tracking.baseline import *
from .file import *
from .fitting.models import *
from .fitting.fit import FdFit
from .fitting.parameter_trace import parameter_trace
from .nb_widgets.range_selector import FdRangeSelector, FdDistanceRangeSelector
from .kymotracker.kymotracker import *
from .nb_widgets.kymotracker_widgets import KymoWidgetGreedy
from .fdensemble import FdEnsemble
from .population import *
from .force_calibration.convenience import calibrate_force
from .force_calibration.calibration_models import (
    PassiveCalibrationModel,
    ActiveCalibrationModel,
    viscosity_of_water,
)
from .force_calibration.power_spectrum_calibration import (
    calculate_power_spectrum,
    fit_power_spectrum,
)


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
