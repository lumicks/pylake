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
.. [5] Nørrelykke, Simon F., and Henrik Flyvbjerg. "Power spectrum analysis with
       least-squares fitting: amplitude bias and its elimination, with application
       to optical tweezers and atomic force microscope cantilevers." Review of
       Scientific Instruments 81.7 (2010).
.. [6] Schäffer, E., Nørrelykke, S. F., & Howard, J. "Surface forces and drag
       coefficients of microspheres near a plane surface measured with optical
       tweezers." Langmuir, 23(7), 3654-3665 (2007).
"""

import math
from collections import namedtuple

import numpy as np
import scipy
from tabulate import tabulate

from lumicks.pylake.force_calibration.power_spectrum import PowerSpectrum
from lumicks.pylake.force_calibration.detail.power_models import (
    ScaledModel,
    fit_analytical_lorentzian,
)

CalibrationParameter = namedtuple("CalibrationParameter", ["description", "value", "unit"])


class CalibrationResults:
    """Power spectrum calibration results.

    Attributes
    ----------
    model : lumicks.pylake.force_calibration.power_spectrum_calibration.CalibrationModel
        Model used for calibration.
    ps_model : PowerSpectrum
        Power spectrum of the fitted model.
    ps_data : PowerSpectrum
        Power spectrum of the data that the model was fitted to.
    params : dict
        Dictionary of input parameters.
    results : dict
        Dictionary of calibration results.
    """

    def __init__(self, model, ps_model, ps_data, params, results):
        self.model = model
        self.ps_model = ps_model
        self.ps_data = ps_data
        self.params = params
        self.results = results

        # A few parameters have to be present for this calibration to be used.
        mandatory_params = ["kappa", "Rf"]
        for key in mandatory_params:
            if key not in results:
                raise RuntimeError(f"Calibration did not provide calibration parameter {key}")

    def __contains__(self, key):
        return key in self.params or key in self.results

    def __getitem__(self, item):
        return self.params[item] if item in self.params else self.results[item]

    def plot(self):
        """Plot the fitted spectrum"""
        import matplotlib.pyplot as plt

        self.ps_data.plot(label="Data")
        self.ps_model.plot(label="Model")
        plt.legend()

    def plot_spectrum_residual(self):
        """Plot the residuals of the fitted spectrum.

        This diagnostic plot can be used to determine how well the spectrum fits the data. While
        it cannot be used to diagnose over-fitting (being unable to reliably estimate parameters
        due to insufficient information in the data), it can be used to diagnose under-fitting (the
        model not fitting the data adequately).

        In an ideal situation, the residual plot should show a noise band around 1 without any
        systematic deviations.
        """
        import matplotlib.pyplot as plt

        residual = self.ps_data.power / self.ps_model.power
        theoretical_std = 1.0 / np.sqrt(self.ps_model.num_points_per_block)

        plt.plot(self.ps_data.frequency, residual, ".")
        plt.axhline(1.0 + theoretical_std, color="k", linestyle="--")
        plt.axhline(1.0 - theoretical_std, color="k", linestyle="--")
        plt.ylabel("Data / Fit [-]")
        plt.xlabel("Frequency [Hz]")

    def _print_data(self, tablefmt="text"):
        def generate_table(entries):
            return [
                [
                    key,
                    f"{param.description}{f' ({param.unit})' if param.unit else ''}",
                    param.value
                    if isinstance(param.value, str)
                    else ("" if param.value is None else f"{param.value:.6g}"),
                ]
                for key, param in entries.items()
            ]

        return tabulate(
            generate_table(self.params) + generate_table(self.results),
            ["Name", "Description", "Value"],
            tablefmt=tablefmt,
        )

    def _repr_html_(self):
        return self._print_data(tablefmt="html")

    def __str__(self) -> str:
        return self._print_data()


def calculate_power_spectrum(
    data, sample_rate, fit_range=(1e2, 23e3), num_points_per_block=2000, excluded_ranges=None
) -> PowerSpectrum:
    """Compute power spectrum and return it as a
    :class:`~lumicks.pylake.force_calibration.power_spectrum.PowerSpectrum`.

    Parameters
    ----------
    data : np.array
        Data used for calibration.
    sample_rate : float
        Sampling rate [Hz]
    fit_range : tuple of float, optional
        Tuple of two floats (f_min, f_max), indicating the frequency range to use for the
        full model fit. [Hz]
    num_points_per_block : int, optional
        The spectrum is first block averaged by this number of points per block.
        Default: 2000.
    excluded_ranges : list of tuple of float, optional
        List of ranges to exclude specified as a list of (frequency_min, frequency_max).

    Returns
    -------
    PowerSpectrum

    Raises
    ------
    TypeError
        If the data is not a one-dimensional numpy array.
    """
    if not isinstance(data, np.ndarray) or (data.ndim != 1):
        raise TypeError('Argument "data" must be a numpy vector')

    power_spectrum = PowerSpectrum(data, sample_rate)
    power_spectrum = power_spectrum.in_range(*fit_range)._exclude_range(excluded_ranges)
    power_spectrum = power_spectrum.downsampled_by(num_points_per_block)

    return power_spectrum


def lorentzian_loss(p, model: ScaledModel, power_spectrum: PowerSpectrum):
    expectation = model(power_spectrum.frequency, p[0], p[1], *p[2:])
    gamma = expectation / power_spectrum.num_points_per_block**0.5
    # fmt: off
    return np.sum(np.log(
        1 + 0.5 * ((power_spectrum.power - model(power_spectrum.frequency, p[0], p[1], *p[2:]))
                   / gamma) ** 2
    ))
    # fmt: on


def fit_power_spectrum(
    power_spectrum,
    model,
    analytical_fit_range=(1e1, 1e4),
    ftol=1e-7,
    max_function_evals=10000,
    bias_correction=True,
    loss_function="gaussian",
) -> CalibrationResults:
    """Fit a power spectrum.

    Performs force calibration. The power spectrum calibration algorithms implemented here are
    based on [1]_ [2]_ [3]_ [4]_ [5]_ [6]_.

    Parameters
    ----------
    power_spectrum : PowerSpectrum
        A power spectrum used for calibration
    model : CalibrationModel
        The model to be used for power spectrum calibration.
    analytical_fit_range : tuple (f_min, f_max), optional
        Tuple of two floats, indicating the frequency range to use for the
        analytical simple Lorentzian fit, used to obtain initial parameter
        guesses [Hz]
    ftol : float
        Termination tolerance for the model fit.
    max_function_evals : int
        Maximum number of function evaluations during the fit.
    bias_correction : bool
        Apply bias correction to the estimate of the diffusion coefficient according to [5]. This
        bias correction is calculated as N/(N+1) where N represents the number of points used in the
        computation of each data point in the power spectral density. Valid only for
        loss_function == "gaussian" (see below).
    loss_function : string
        Loss function to use during fitting. Options: "gaussian" (default),
        "lorentzian". For least squares fitting, use "gaussian". For robust fitting, that is, an
        attempt to ignore spurious peaks in the power spectrum, use "lorentzian". In that case, no
        error estimates are available, and the corresponding entries are set to NaN.
        This is beta functionality. While usable, this has not yet been tested in a large
        number of different scenarios. The API can still be subject to change without any prior
        deprecation notice! If you use this functionality keep a close eye on the changelog for any
        changes that may affect your analysis.

    Returns
    -------
    CalibrationResults
        Parameters obtained from the calibration procedure.

    Raises
    ------
    RuntimeError
        If there are fewer than 4 data points to fit in the power spectrum.
    TypeError
        If the supplied power spectrum is not of the type `PowerSpectrum`.
    RuntimeError
        If there is insufficient data to perform the analytical fit used as initial condition.
    ValueError
        If the loss function is not one of the possible values (case sensitive)
    Runtime Error
        If bias correction is on and the loss function is not "gaussian"

    References
    ----------
    .. [1] Berg-Sørensen, K. & Flyvbjerg, H. Power spectrum analysis for optical tweezers. Rev. Sci.
           Instrum. 75, 594 (2004).
    .. [2] Tolić-Nørrelykke, I. M., Berg-Sørensen, K. & Flyvbjerg, H. MatLab program for precision
           calibration of optical tweezers. Comput. Phys. Commun. 159, 225–240 (2004).
    .. [3] Hansen, P. M., Tolic-Nørrelykke, I. M., Flyvbjerg, H. & Berg-Sørensen, K.
           tweezercalib 2.1: Faster version of MatLab package for precise calibration of optical
           tweezers. Comput. Phys. Commun. 175, 572–573 (2006).
    .. [4] Berg-Sørensen, K., Peterman, E. J. G., Weber, T., Schmidt, C. F. & Flyvbjerg, H. Power
           spectrum analysis for optical tweezers. II: Laser wavelength dependence of parasitic
           filtering, and how to achieve high bandwidth. Rev. Sci. Instrum. 77, 063106 (2006).
    .. [5] Tolić-Nørrelykke, S. F, and Flyvbjerg, H, "Power spectrum analysis with least-squares
           fitting: amplitude bias and its elimination, with application to optical tweezers and
           atomic force microscope cantilevers." Review of Scientific Instruments 81.7 (2010)
    .. [6] Tolić-Nørrelykke S. F, Schäffer E, Howard J, Pavone F. S, Jülicher F and Flyvbjerg, H.
           Calibration of optical tweezers with positional detection in the back focal plane,
           Review of scientific instruments 77, 103101 (2006).
    """
    if len(power_spectrum.frequency) < 4:
        raise RuntimeError(
            "Insufficient number of points to fit power spectrum. Check whether"
            "you are using the correct frequency range and sampling rate."
        )
    if not isinstance(power_spectrum, PowerSpectrum):
        raise TypeError('Argument "power_spectrum" must be of type PowerSpectrum')

    if loss_function not in ["gaussian", "lorentzian"]:
        raise ValueError('Argument "loss_function" must be "gaussian" or "lorentzian"')

    if bias_correction and loss_function == "lorentzian":
        raise RuntimeError('Bias correction and loss function="lorentzian" are mutually exclusive')

    # Fit analytical Lorentzian to get initial guesses for the full power spectrum model.
    analytical_power_spectrum = power_spectrum.in_range(*analytical_fit_range)
    if len(analytical_power_spectrum.frequency) < 1:
        raise RuntimeError(
            "An empty power spectrum was passed to fit_analytical_lorentzian. Check"
            "whether you are using the correct sample rate and frequency range"
            "for the analytical fit."
        )
    anl_fit_res = fit_analytical_lorentzian(analytical_power_spectrum)

    initial_params = np.array([anl_fit_res.fc, anl_fit_res.D, *model._filter.initial_values])

    # The actual curve fitting process is driven by a set of fit parameters that are of order unity.
    # This increases the robustness of the fit (see ref. 3). The `ScaledModel` model class takes
    # care of this parameter rescaling.
    scaled_model = ScaledModel(
        lambda f, fc, D, *filter_pars: 1 / model(f, fc, D, *filter_pars), initial_params
    )

    # What we *actually* have to minimize, is the chi^2 expression in Eq. 39 of ref. 1. We're
    # "hacking" `scipy.optimize.curve_fit` for this purpose, and passing in "y" values of "1/ps.power"
    # instead of just "ps.power", and a model function "1/model(...)" instead of "model(...)". This
    # effectively transforms the curve fitter's objective function
    # "np.sum( ((f(xdata, *popt) - ydata) / sigma)**2 )" into the expression in Eq. 39 of ref. 1.
    sigma = (1 / power_spectrum.power) / math.sqrt(power_spectrum.num_points_per_block)
    lower_bounds = scaled_model.normalize_params([0.0, 0.0, *model._filter.lower_bounds()])
    upper_bounds = scaled_model.normalize_params(
        [np.inf, np.inf, *model._filter.upper_bounds(power_spectrum.sample_rate)]
    )
    (solution_params_rescaled, pcov) = scipy.optimize.curve_fit(
        scaled_model,
        power_spectrum.frequency,
        1 / power_spectrum.power,
        p0=np.ones(len(initial_params)),
        sigma=sigma,
        absolute_sigma=True,
        method="trf",
        ftol=ftol,
        maxfev=max_function_evals,
        bounds=(lower_bounds, upper_bounds),
    )
    solution_params_rescaled = np.abs(solution_params_rescaled)
    perr = np.sqrt(np.diag(pcov))

    # Undo the scaling
    solution_params = scaled_model.scale_params(solution_params_rescaled)
    perr = scaled_model.scale_params(perr)

    # Use the least squares method as a starting point for a robust fit
    if loss_function == "lorentzian":
        scaled_model = ScaledModel(
            lambda f, fc, D, *filter_pars: model(f, fc, D, *filter_pars), solution_params
        )
        bounds = [(lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)]
        minimize_result = scipy.optimize.minimize(
            lambda p: lorentzian_loss(p, scaled_model, power_spectrum),
            np.ones(len(solution_params)),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_function_evals, "ftol": ftol},
        )
        solution_params_scaled = np.abs(minimize_result.x)
        solution_params = scaled_model.scale_params(solution_params_scaled)

        # Return NaN for perr, as scipy.optimize.minimize does not return the covariance matrix
        perr = np.NaN * np.ones_like(solution_params)

    # Calculate goodness-of-fit, in terms of the statistical backing (see ref. 1).
    chi_squared = np.sum(
        ((1 / model(power_spectrum.frequency, *solution_params) - 1 / power_spectrum.power) / sigma)
        ** 2
    )
    n_degrees_of_freedom = power_spectrum.power.size - len(solution_params)
    chi_squared_per_deg = chi_squared / n_degrees_of_freedom
    backing = (1 - scipy.special.gammainc(chi_squared / 2, n_degrees_of_freedom / 2)) * 100

    # Fitted power spectrum values.
    ps_model = power_spectrum.with_spectrum(
        model(power_spectrum.frequency, *solution_params), power_spectrum.num_points_per_block
    )

    # When using theoretical weights for fitting, ref [5] mentions that the found value for D will
    # be biased by a factor (n+1)/n. Multiplying by n/(n+1) compensates for this.
    if bias_correction:
        bias_corr = power_spectrum.num_points_per_block / (power_spectrum.num_points_per_block + 1)
        solution_params[1] *= bias_corr
        perr[1] *= bias_corr

    return CalibrationResults(
        model=model,
        ps_data=power_spectrum,
        ps_model=ps_model,
        results={
            **model.calibration_results(
                fc=solution_params[0],
                diffusion_constant_volts=solution_params[1],
                filter_params=solution_params[2:],
                fc_err=perr[0],
                diffusion_constant_volts_err=perr[1],
                filter_params_err=perr[2:],
            ),
            "chi_squared_per_deg": CalibrationParameter(
                "Chi squared per degree of freedom", chi_squared_per_deg, ""
            ),
            "backing": CalibrationParameter("Statistical backing", backing, "%"),
        },
        params={
            **model.calibration_parameters(),
            "Max iterations": CalibrationParameter(
                "Maximum number of function evaluations", max_function_evals, ""
            ),
            "Fit tolerance": CalibrationParameter("Fitting tolerance", ftol, ""),
            "Points per block": CalibrationParameter(
                "Number of points per block", power_spectrum.num_points_per_block, ""
            ),
            "Sample rate": CalibrationParameter("Sample rate", power_spectrum.sample_rate, "Hz"),
            "Bias correction": CalibrationParameter(
                "Perform bias correction thermal fit", bias_correction, ""
            ),
            "Loss function": CalibrationParameter(
                "Loss function used during minimization", loss_function, ""
            ),
        },
    )
