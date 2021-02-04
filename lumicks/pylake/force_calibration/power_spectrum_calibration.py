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
from lumicks.pylake.force_calibration.detail.power_spectrum import PowerSpectrum
from lumicks.pylake.force_calibration.detail.power_models import (
    fit_analytical_lorentzian,
    FullPSFitModel,
    _a,
    _alpha,
    sphere_friction_coefficient,
)


class CalibrationParameters:
    """Power spectrum calibration parameters

    Attributes
    ----------
    bead_diameter : float
        Bead diameter [um].
    viscosity : float, optional
        Liquid viscosity [Pa*s]. Default: 1.002e-3 Pa*s.
    temperature : float, optional
        Liquid temperature [Celsius].
    """

    def __init__(self, bead_diameter, **kwargs):
        self.bead_diameter = bead_diameter
        self.viscosity = 1.002e-3
        self.temperature = 20

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise TypeError("Unknown argument %s" % k)

    def temperature_K(self):
        return scipy.constants.convert_temperature(self.temperature, "C", "K")


class CalibrationSettings:
    """Power spectrum calibration algorithm settings

    Attributes
    ----------
    n_points_per_block : int, optional
        The data in `ps` is first blocked, with approximately this number of
        points per block. Default: 350.
    fit_range : tuple (f_min, f_max), optional
        Tuple of two floats, indicating the frequency range to use for the
        full model fit. Default: (1e2, 23e3) [Hz]
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
        self.n_points_per_block = 350
        self.fit_range = (1e2, 23e3)
        self.analytical_fit_range = (1e1, 1e4)
        self.ftol = 1e-7
        self.maxfev = 10000

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise TypeError("Unknown argument %s" % k)


class CalibrationResults:
    """Power spectrum calibration results

    Attributes
    ----------
    fc : float
        Corner frequency [Hz]
    D : float
        Diffusion constant [V^2/s]
    f_diode : float
        Diode low-pass filtering roll-off frequency [Hz]
    alpha : float
        Diode 'relaxation factor' (number between 0 and 1)
    err_fc : float
        1-sigma error for the parameters fc
    err_D
    err_f_diode
    err_alpha
    Rd : float
        Distance response [um/V]
    kappa : float
        Trap stiffness [pN/nm]
    Rf : float
        Force response [pN/V]
    chi_squared_per_deg : float
        Chi-squared per degree of freedom
    backing : float
        Statistical backing [%]
    ps_fitted : PowerSpectrum
        Power spectrum that was actually fitted (filtered and block-averaged)
    ps_model_fit : PowerSpectrum
        Model fit to the power spectrum (for, e.g., plotting)
    error : str
        Optional error message, in case problems were encountered during the
        fit.

    NOTE: Any of the above attributes can be absent, in case of errors. In such
    cases, the ``error`` attribute typically explains what went wrong.
    """

    def __init__(self, **kwargs):
        _valid_attr = [
            "fc",
            "D",
            "f_diode",
            "alpha",
            "err_fc",
            "err_D",
            "err_f_diode",
            "err_alpha",
            "chi_squared_per_deg",
            "backing",
            "ps_fitted",
            "ps_model_fit",
            "Rd",
            "kappa",
            "Rf",
            "error",
        ]

        for k, v in kwargs.items():
            if k in _valid_attr:
                setattr(self, k, v)
            else:
                raise TypeError("Unknown argument/attribute %s" % k)

    def is_success(self):
        return not hasattr(self, "error")


class CalibrationError(Exception):
    pass


class PowerSpectrumCalibration:
    """Power Spectrum Calibration

    Attributes
    ----------
    ps : PowerSpectrum
        Power spectrum calculated from input data
    params : CalibrationParameters
        Calibration parameters.
    settings : CalibrationSettings
        Calibration algorithm settings.
    results : CalibrationResults
        Any results from the calibration. Can be `None`.
    """

    def __init__(self, data, sampling_rate, params, settings=None):
        if not isinstance(data, np.ndarray) or (data.ndim != 1):
            raise TypeError('Argument "data" must be a numpy vector')
        self.ps = PowerSpectrum(data, sampling_rate)

        if not isinstance(params, CalibrationParameters):
            raise TypeError('Argument "params" must be of type CalibrationParameters')
        self.params = params

        if settings:
            if not isinstance(settings, CalibrationSettings):
                raise TypeError('Argument "settings" must be of type CalibrationSettings')
            self.settings = settings
        else:
            self.settings = CalibrationSettings()
        self.results = None

    @staticmethod
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
        f_nyquist = ps.sampling_rate / 2
        P_aliased_nyq = (guess_D / (2 * math.pi ** 2)) / (f_nyquist ** 2 + guess_fc ** 2)
        if ps.P[-1] < P_aliased_nyq:
            dif = ps.P[-1] / P_aliased_nyq
            return math.sqrt(dif * f_nyquist ** 2 / (1.0 - dif))
        else:
            return 2 * f_nyquist

    def run_fit(self, print_diagnostics=False):
        """Runs the actual fitting procedure

        Parameters
        ----------
        print_diagnostics : bool
            If True, prints diagnostics about the fitting procedure to STDOUT.
        """
        # Filter and block the power spectrum.
        ps = self.ps.in_range(*self.settings.fit_range)
        ps = ps.block_averaged(n_blocks=ps.P.size // self.settings.n_points_per_block)

        try:
            # First do an analytical simple Lorentzian fit, to get some initial
            # parameter guesses for the fit.
            anl_fit_ps = ps.in_range(*self.settings.analytical_fit_range)
            anl_fit_res = fit_analytical_lorentzian(anl_fit_ps)
            if not anl_fit_res:
                raise ValueError("Analytical fit failed")
            initial_params = (
                anl_fit_res.fc,
                anl_fit_res.D,
                self.guess_f_diode_initial_value(ps, anl_fit_res.fc, anl_fit_res.D),
                0.3,
            )

            if print_diagnostics:
                print("Initial fit parameters:   fc = %.2e  D = %.2f  f_diode = %.2e  alpha = %.2f" % initial_params)
            # Then do a Levenberg-Marquardt weighted least-squares fit on the full model.
            #
            # Technical notes:
            #
            # - Instead of directly fitting the model parameter alpha (a characteristic
            #   of the PSD diode), we instead plug a transformed variable "a = sqrt(1/alpha^2-1)"
            #   into the optimization process. In the model, we then transform "a" back
            #   using "alpha = 1/sqrt(1+a^2)". The latter function is bounded between
            #   [0,1], so we have effectively created a bound constraint on alpha.
            #
            # - The actual curve fitting process is driven by a set of fit parameters
            #   that are of order unity. This increases the robustness of the fit
            #   (see ref. 3). The "FullPSFitModel" model class takes care of this
            #   parameter rescaling.
            #
            # - What we *actually* have to minimize, is the chi^2 expression in Eq. 39
            #   of ref. 1. We're "hacking" `scipy.optimize.curve_fit` for this purpose,
            #   and passing in "y" values of "1/ps.P" instead of just "ps.P",
            #   and a model function "1/model(...)" instead of "model(...)". This
            #   effectively transforms the curve fitter's objective function
            #   "np.sum( ((f(xdata, *popt) - ydata) / sigma)**2 )" into the expression
            #   in Eq. 39 of ref. 1.

            model = FullPSFitModel(
                scale_factors=(
                    *initial_params[0:3],
                    _a(initial_params[3]),
                )
            )
            sigma = (1 / ps.P) / math.sqrt(self.settings.n_points_per_block)
            (solution_params_rescaled, pcov) = scipy.optimize.curve_fit(
                lambda f, fc, D, f_diode, a: 1 / model(f, fc, D, f_diode, a),
                ps.f,
                1 / ps.P,
                p0=np.ones(4),
                sigma=sigma,
                absolute_sigma=True,
                method="lm",
                ftol=self.settings.ftol,
                maxfev=self.settings.maxfev,
            )
            solution_params_rescaled = np.abs(solution_params_rescaled)
            # the model function is symmetric in alpha and f_diode...
            solution_params = model.get_params_from_rescaled_params(solution_params_rescaled)

            # Calculate goodness-of-fit, in terms of the statistical backing (see ref. 1).
            chi_squared = np.sum(((1 / model(ps.f, *solution_params_rescaled) - 1 / ps.P) / sigma) ** 2)
            n_degrees_of_freedom = ps.P.size - len(solution_params)
            chi_squared_per_deg = chi_squared / n_degrees_of_freedom
            backing = (1 - scipy.special.gammainc(chi_squared / 2, n_degrees_of_freedom / 2)) * 100

            # We also have to un-rescale the covariance matrix.
            # There's actually a rescaling factor *squared* in there, as the Jacobian
            # appears twice in the equation for the covariance matrix.
            perr = np.sqrt(np.diag(pcov) * (np.array(model.scale_factors) ** 2))

            # TODO Fix calculation of alpha confidence interval.
            # The previous step calculated the confidence interval in the transformed
            # variable 'a', *not* in 'alpha'! We're using this rather ugly, most likely
            # not-quite-statistically-correct trick to transform the 'a' confidence
            # interval into an 'alpha' confidence interval. Note that this also seems
            # to give us different results for the alpha confidence interval than the
            # original tweezercalib-2.1 code from ref. 3.
            perr[3] = (
                abs(
                    _alpha(solution_params_rescaled[3] * model.scale_factors[3] + perr[3])
                    - _alpha(solution_params_rescaled[3] * model.scale_factors[3] - perr[3])
                )
                / 2
            )

            # Fitted power spectrum values.
            ps_model_fit = PowerSpectrum()
            ps_model_fit.f = ps.f
            ps_model_fit.P = model.P(ps.f, *solution_params)
            ps_model_fit.sampling_rate = ps.sampling_rate
            ps_model_fit.T_measure = ps.T_measure

            if print_diagnostics:
                print("Solution:   fc = %.2e  D = %.2f  f_diode = %.2e  alpha = %.2f" % solution_params)
                print("Errors:     fc = %.2e  D = %.2f  f_diode = %.2e  alpha = %.2f" % tuple(perr))
                print("Units:      fc : Hz        D : V^2/s f_diode : Hz")
                print("Chi^2 per degree of freedom = %.2f" % chi_squared_per_deg)
                print("Statistical backing = %.1f%%" % backing)
            # Calculate additional calibration constants.
            gamma_0 = sphere_friction_coefficient(self.params.viscosity, self.params.bead_diameter * 1e-6)
            Rd = math.sqrt(scipy.constants.k * self.params.temperature_K() / gamma_0 / solution_params[1]) * 1e6
            kappa = 2 * math.pi * gamma_0 * solution_params[0] * 1e3
            Rf = Rd * kappa * 1e3

            # Return fit results.
            self.results = CalibrationResults(
                fc=solution_params[0],
                D=solution_params[1],
                f_diode=solution_params[2],
                alpha=solution_params[3],
                err_fc=perr[0],
                err_D=perr[1],
                err_f_diode=perr[2],
                err_alpha=perr[3],
                chi_squared_per_deg=chi_squared_per_deg,
                backing=backing,
                ps_fitted=ps,
                ps_model_fit=ps_model_fit,
                Rd=Rd,
                kappa=kappa,
                Rf=Rf,
            )
        except (ValueError, RuntimeError) as e:
            self.results = CalibrationResults(ps_fitted=ps)
            raise CalibrationError(str(e))
