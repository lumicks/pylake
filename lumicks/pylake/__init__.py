from . import simulation
from .file import *
from .scalebar import ScaleBar
from .__about__ import (
    __doc__,
    __url__,
    __email__,
    __title__,
    __author__,
    __license__,
    __summary__,
    __version__,
    __copyright__,
)
from .benchmark import benchmark
from .fdensemble import FdEnsemble
from .population import *
from .adjustments import ColorAdjustment, colormaps
from .fitting.fit import FdFit
from .image_stack import ImageStack, CorrelatedStack
from .file_download import *
from .fitting.models import *
from .fitting.parameter_trace import parameter_trace
from .kymotracker.kymotracker import *
from .piezo_tracking.baseline import *
from .nb_widgets.range_selector import FdRangeSelector, FdDistanceRangeSelector
from .force_calibration.convenience import calibrate_force
from .piezo_tracking.piezo_tracking import *
from .nb_widgets.kymotracker_widgets import KymoWidgetGreedy
from .force_calibration.calibration_models import (
    ActiveCalibrationModel,
    PassiveCalibrationModel,
    density_of_water,
    viscosity_of_water,
)
from .force_calibration.power_spectrum_calibration import (
    fit_power_spectrum,
    calculate_power_spectrum,
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
    import pathlib

    import pytest

    args = args or []
    module_path = str(pathlib.Path(__file__).parent)
    args.append(module_path)

    error_code = pytest.main(args, plugins)
    return error_code or None
