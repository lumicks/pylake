"""Functions for Power Spectrum calibration of optical tweezers.

Notes
-----
The power spectrum calibration algorithm implemented here is based on a number
of publications by the Flyvbjerg group at DTU [1]_ [2]_ [3]_ [4]_.

References
----------
.. [1] Berg-Sørensen, K. & Flyvbjerg, H. Power spectrum analysis for optical
       tweezers. Rev. Sci. Instrum. 75, 594 (2004).
.. [2] Tolić-Nørrelykke, I. M., Berg-Sørensen, K. & Flyvbjerg, H. MatLab program
       for precision calibration of optical tweezers. Comput. Phys. Commun. 159,
       225–240 (2004).
.. [3] Hansen, P. M., Tolic-Nørrelykke, I. M., Flyvbjerg, H. & Berg-Sørensen, K.
       tweezercalib 2.1: Faster version of MatLab package for precise calibration
       of optical tweezers. Comput. Phys. Commun. 175, 572–573 (2006).
.. [4] Berg-Sørensen, K., Peterman, E. J. G., Weber, T., Schmidt, C. F. &
       Flyvbjerg, H. Power spectrum analysis for optical tweezers. II: Laser
       wavelength dependence of parasitic filtering, and how to achieve high
       bandwidth. Rev. Sci. Instrum. 77, 063106 (2006).
"""

import numpy as np
import math
import scipy
import scipy.optimize
import scipy.constants
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import namedtuple
from lumicks.pylake.force_calibration.power_spectrum import PowerSpectrum
from lumicks.pylake.force_calibration.detail.power_models import (
    ScaledModel,
    fit_analytical_lorentzian,
)


class CalibrationSettings:
    """Power spectrum calibration algorithm settings

    Attributes
    ----------
    analytical_fit_range : tuple (f_min, f_max), optional
        Tuple of two floats, indicating the frequency range to use for the
        analytical simple Lorentzian fit, used to obtain initial parameter
        guesses. Default: (1e1, 1e4) [Hz]
    ftol : float
        Termination tolerance for the model fit. Default: 1e-7
    maxfev : int
        Maximum number of function evaluations during the fit. Default: 10000
    """

    def __init__(self, **kwargs):
        self.analytical_fit_range = (1e1, 1e4)
        self.ftol = 1e-7
        self.maxfev = 10000

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise TypeError("Unknown argument %s" % k)


CalibrationParameter = namedtuple("CalibrationParameter", ["description", "value", "unit"])


class CalibrationResults:
    """Power spectrum calibration results."""

    def __init__(self, model, ps_model_fit, ps_fitted, params):
        self.model = model
        self.ps_model_fit = ps_model_fit
        self.ps_fitted = ps_fitted
        self.params = params

        # A few parameters have to be present for this calibration to be used.
        mandatory_params = ["kappa (pN/nm)", "Rd (um/V)", "Rf (pN/V)"]
        for key in mandatory_params:
            if key not in params:
                raise RuntimeError(f"Calibration did not provide calibration parameter {key}")

    def __getitem__(self, item):
        # Provides the same access pattern as ForceCalibration
        return self.params[item].value

    def plot(self):
        """Plot the fitted spectrum"""
        self.ps_model_fit.plot()
        self.ps_fitted.plot()
        plt.legend(["Data", "Model"])

    def _print_data(self, tablefmt="text"):
        table = [
            [
                key,
                f"{param.description} ({param.unit})",
                param.value if isinstance(param.value, str) else f"{param.value:.6g}",
            ]
            for key, param in self.params.items()
        ]

        return tabulate(table, ["Name", "Description", "Value"], tablefmt=tablefmt)

    def _repr_html_(self):
        return self._print_data(tablefmt="html")

    def __str__(self):
        return self._print_data()


def guess_f_diode_initial_value(ps, guess_fc, guess_D):
    """Calculates a good initial guess for the fit parameter `f_diode`.

    Parameters
    ----------
    ps : PowerSpectrum
        Power spectrum data, as will be passed into the `fit_full_powerspectrum`
        function.
    guess_fc : float
        Guess for the corner frequency, in Hz.
    guess_D : float
        Guess for the diffusion constant, in (a.u.)^2/s.

    Returns
    -------
    float:
        A good initial value for the parameter `f_diode`, for fitting the full
        power spectrum.
    """
    f_nyquist = ps.sample_rate / 2
    P_aliased_nyq = (guess_D / (2 * math.pi ** 2)) / (f_nyquist ** 2 + guess_fc ** 2)
    if ps.power[-1] < P_aliased_nyq:
        dif = ps.power[-1] / P_aliased_nyq
        return math.sqrt(dif * f_nyquist ** 2 / (1.0 - dif))
    else:
        return 2 * f_nyquist


def calculate_power_spectrum(data, sample_rate, fit_range=(1e2, 23e3), num_points_per_block=2000):
    """Compute power spectrum and returns it as a :class:`~.PowerSpectrum`.

    Parameters
    ----------
    data : np.array
        Data used for calibration.
    sample_rate : float
        Sampling rate [Hz]
    fit_range : tuple (f_min, f_max), optional
        Tuple of two floats, indicating the frequency range to use for the
        full model fit. Default: (1e2, 23e3) [Hz]
    num_points_per_block : int, optional
        The spectrum is first block averaged by this number of points per block.
        Default: 2000.

    Returns
    -------
    :class:`~.PowerSpectrum`
        Estimated power spectrum based.
    """
    if not isinstance(data, np.ndarray) or (data.ndim != 1):
        raise TypeError('Argument "data" must be a numpy vector')

    power_spectrum = PowerSpectrum(data, sample_rate)
    power_spectrum = power_spectrum.in_range(*fit_range)
    power_spectrum = power_spectrum.downsampled_by(num_points_per_block)

    return power_spectrum


def fit_power_spectrum(power_spectrum, model, settings=CalibrationSettings()):
    """Power Spectrum Calibration

    Parameters
    ----------
    power_spectrum : PowerSpectrum
        A power spectrum used for calibration
    model : CalibrationModel
        The model to be used for power spectrum calibration.
    settings : CalibrationSettings
        Calibration algorithm settings.

    Returns
    -------
    :class:`~.CalibrationResults`
        Parameters obtained from the calibration procedure.
    """
    if len(power_spectrum.frequency) < 4:
        raise RuntimeError(
            "Insufficient number of points to fit power spectrum. Check whether"
            "you are using the correct frequency range and sampling rate."
        )
    if not isinstance(power_spectrum, PowerSpectrum):
        raise TypeError('Argument "power_spectrum" must be of type PowerSpectrum')
    if not isinstance(settings, CalibrationSettings):
        raise TypeError('Argument "settings" must be of type CalibrationSettings')

    # Fit analytical Lorentzian to get initial guesses for the full power spectrum model.
    analytical_power_spectrum = power_spectrum.in_range(*settings.analytical_fit_range)
    if len(analytical_power_spectrum.frequency) < 1:
        raise RuntimeError(
            "An empty power spectrum was passed to fit_analytical_lorentzian. Check"
            "whether you are using the correct sample rate and frequency range"
            "for the analytical fit."
        )
    anl_fit_res = fit_analytical_lorentzian(analytical_power_spectrum)

    initial_params = (
        anl_fit_res.fc,
        anl_fit_res.D,
        guess_f_diode_initial_value(power_spectrum, anl_fit_res.fc, anl_fit_res.D),
        0.3,
    )

    # Instead of directly fitting the model parameter alpha (a characteristic of the PSD diode),
    # we instead plug a transformed variable "a = sqrt(1/alpha^2-1)" into the optimization process.
    # In the model, we then transform "a" back using "alpha = 1/sqrt(1+a^2)". The latter function
    # is bounded between [0,1], so we have effectively created a bound constraint on alpha.
    #
    # The actual curve fitting process is driven by a set of fit parameters that are of order unity.
    # This increases the robustness of the fit (see ref. 3). The `ScaledModel` model class takes
    # care of this parameter rescaling.
    scaled_model = ScaledModel(model, initial_params)

    # What we *actually* have to minimize, is the chi^2 expression in Eq. 39 of ref. 1. We're
    # "hacking" `scipy.optimize.curve_fit` for this purpose, and passing in "y" values of "1/ps.power"
    # instead of just "ps.power", and a model function "1/model(...)" instead of "model(...)". This
    # effectively transforms the curve fitter's objective function
    # "np.sum( ((f(xdata, *popt) - ydata) / sigma)**2 )" into the expression in Eq. 39 of ref. 1.
    sigma = (1 / power_spectrum.power) / math.sqrt(power_spectrum.num_points_per_block)
    (solution_params_rescaled, pcov) = scipy.optimize.curve_fit(
        lambda f, fc, D, f_diode, a: 1 / scaled_model(f, fc, D, f_diode, a),
        power_spectrum.frequency,
        1 / power_spectrum.power,
        p0=np.ones(4),
        sigma=sigma,
        absolute_sigma=True,
        method="lm",
        ftol=settings.ftol,
        maxfev=settings.maxfev,
    )
    solution_params_rescaled = np.abs(solution_params_rescaled)
    perr = np.sqrt(np.diag(pcov))

    # Undo the scaling
    solution_params = scaled_model.scale_params(solution_params_rescaled)
    perr = scaled_model.scale_stderrs(solution_params_rescaled, perr)

    # Calculate goodness-of-fit, in terms of the statistical backing (see ref. 1).
    chi_squared = np.sum(
        ((1 / model(power_spectrum.frequency, *solution_params) - 1 / power_spectrum.power) / sigma)
        ** 2
    )
    n_degrees_of_freedom = power_spectrum.power.size - len(solution_params)
    chi_squared_per_deg = chi_squared / n_degrees_of_freedom
    backing = (1 - scipy.special.gammainc(chi_squared / 2, n_degrees_of_freedom / 2)) * 100

    # Fitted power spectrum values.
    ps_model_fit = power_spectrum.with_spectrum(model(power_spectrum.frequency, *solution_params))

    return CalibrationResults(
        model=model,
        ps_fitted=power_spectrum,
        ps_model_fit=ps_model_fit,
        params={
            **model.calibration_parameters(
                fc=solution_params[0], diffusion_constant=solution_params[1]
            ),
            "fc (Hz)": CalibrationParameter("Corner frequency", solution_params[0], "Hz"),
            "D (V^2/s)": CalibrationParameter("Diffusion constant", solution_params[1], "V^2/s"),
            "f_diode (Hz)": CalibrationParameter(
                "Diode low-pass filtering roll-off frequency", solution_params[2], "Hz"
            ),
            "alpha": CalibrationParameter("Diode 'relaxation factor'", solution_params[3], "-"),
            "err_fc": CalibrationParameter("Corner frequency Std Err", perr[0], "Hz"),
            "err_D": CalibrationParameter("Diffusion constant Std Err", perr[1], "V^2/s"),
            "err_f_diode": CalibrationParameter(
                "Diode low-pass filtering roll-off frequency Std Err", perr[2], "Hz"
            ),
            "err_alpha": CalibrationParameter("Diode 'relaxation factor' Std Err", perr[3], "-"),
            "chi_squared_per_deg": CalibrationParameter(
                "Chi squared per degree of freedom", chi_squared_per_deg, "-"
            ),
            "backing (%)": CalibrationParameter("Statistical backing", backing, "%"),
            "Max iterations": CalibrationParameter(
                "Maximum number of function evaluations", settings.maxfev, "-"
            ),
            "Model": CalibrationParameter("Calibration model", model.__name__(), "-"),
            "Fit tolerance": CalibrationParameter("Fitting tolerance", settings.ftol, "-"),
            "Points per block": CalibrationParameter(
                "Number of points per block", power_spectrum.num_points_per_block, "-"
            ),
            "Sample rate (Hz)": CalibrationParameter(
                "Sample rate", power_spectrum.sample_rate, "Hz"
            ),
        },
    )
