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
import scipy
import scipy.optimize
import scipy.constants
from math import floor, sqrt, pi, atan
from collections import namedtuple


def block_average(data, n_blocks):
    """Calculates the block average of a dataset.

    For an array ``A`` of length ``N``, returns an array ``B`` of length
    ``M``, where each element of ``B`` is the average of ``q`` neighboring
    elements. ``q`` is equal to ``floor(N/M)``. This implies that if ``N*q``
    is not exactly equal to ``M``, the last partially complete window is
    thrown away by this function.
    """
    block_size = floor(data.size / n_blocks)
    length = block_size * n_blocks
    return np.mean(np.reshape(data[:length], (-1, block_size)), axis=1)


def block_average_std(data, n_blocks):
    """Calculates the block standard deviation of a dataset.

    Works as `block_average`, but calculates the standard deviation
    instead of the average.

    See Also
    --------
    block_average
    """
    block_size = floor(data.size / n_blocks)
    length = block_size * n_blocks
    return np.std(np.reshape(data[:length], (-1, block_size)), axis=1)


class PowerSpectrum:
    """Power spectrum data for a time series.

    Attributes
    ----------
    f : numpy.ndarray
        Frequency values for the power spectrum. [Hz]
    P : numpy.ndarray
        Power values for the power spectrum (typically in V^2/s).
    sampling_rate : float
        The sampling rate for the original data. [Hz]
    T_measure : float
        The total duration of the original data. [seconds]
    """

    def __init__(self, data=None, sampling_rate=None):
        """Constructor

        If neither parameter is given, an empty object is created.

        Parameters
        ----------
        data : numpy.ndarray, optional
            Data from which to calculate a power spectrum.
        sampling_rate : float, optional
        """
        if data is not None:
            # Initialize from raw sensor data.
            assert sampling_rate is not None

            # ... have to subtract average position first.
            data = data - np.mean(data)

            # ... use FFT to calculate power spectrum.
            fft = np.fft.rfft(data)
            self.f = np.fft.rfftfreq(data.size, 1./sampling_rate)
            self.P = (1./sampling_rate) * np.square(np.abs(fft)) / data.size

            # ... store metadata.
            self.sampling_rate = sampling_rate
            self.T_measure = data.size / sampling_rate
        else:
            # Initialize empty object.
            self.f = None
            self.P = None
            self.sampling_rate = None
            self.T_measure = None

    def as_dict(self):
        """"Returns a representation of the PowerSpectrum suitable for serialization"""
        return {'f': self.f.tolist(), 'P': self.P.tolist()}

    def block_averaged(self, n_blocks=2000):
        """Returns a block-averaged power spectrum.

        See Also
        --------
        block_average
        """
        ba = PowerSpectrum()
        ba.f = block_average(self.f, n_blocks)
        ba.P = block_average(self.P, n_blocks)
        ba.sampling_rate = self.sampling_rate
        ba.T_measure = self.T_measure
        return ba

    def in_range(self, f_min, f_max):
        """Returns part of the power spectrum within a given frequency range."""
        ir = PowerSpectrum()
        ir.f = self.f[(self.f > f_min) & (self.f <= f_max)]
        ir.P = self.P[(self.f > f_min) & (self.f <= f_max)]
        ir.sampling_rate = self.sampling_rate
        ir.T_measure = self.T_measure
        return ir

    def n_samples(self):
        return self.f.size


def fit_analytical_lorentzian(ps):
    """Performs an analytical least-squares fit of a Lorentzian Power Spectrum.

    Based on Section IV from ref. 1.

    Parameters
    ----------
    ps : PowerSpectrum
        Power spectrum data. Should generally be block-averaged, before passing
        into this function.

    Returns
    -------
    namedtuple (fc, D, sigma_fc, sigma_D, ps_fit)
        Attributes:
        - `fc` : corner frequency [Hz]
        - `D`: diffusion constant [V^2/s]
        - `sigma_fc`, `sigma_D`: 1-sigma confidence intervals for `fc` and `D`
        - `ps_fit`: `PowerSpectrum` object with model fit

        Note: returns None if the fit fails.
    """
    FitResults = namedtuple('AnalyticalLorentzianFitResults',
                            ['fc', 'D', 'sigma_fc', 'sigma_D', 'ps_fit'])

    # Calculate S[p,q] elements (Ref. 1, Eq. 13-14).
    Spq = np.zeros((3,3))
    for p in range(3):
        for q in range(3):
            Spq[p,q] = np.sum( np.power(ps.f, 2*p) * np.power(ps.P, q) )

    try:
        # Calculate a and b parameters (Ref. 1, Eq. 13-14).
        a = (Spq[0,1] * Spq[2,2] - Spq[1,1] * Spq[1,2]) / (Spq[0,2] * Spq[2,2] - Spq[1,2] * Spq[1,2])
        b = (Spq[1,1] * Spq[0,2] - Spq[0,1] * Spq[1,2]) / (Spq[0,2] * Spq[2,2] - Spq[1,2] * Spq[1,2])

        # Having a and b, calculating fc and D is trivial.
        fc = sqrt(a/b)              # corner frequency [Hz]
        D  = (1/b) * 2 * (pi**2)    # diffusion constant [V^2/s]

        # Fitted power spectrum values.
        ps_fit = PowerSpectrum()
        ps_fit.f = ps.f
        ps_fit.P = 1 / (a + b * np.power(ps.f, 2))
        ps_fit.sampling_rate = ps.sampling_rate
        ps_fit.T_measure = ps.T_measure

        # Error propagation (Ref. 1, Eq. 25-28).
        x_min = ps.f.min() / fc
        x_max = ps.f.max() / fc

        u = (2*x_max)/(1+x_max**2) - (2*x_min)/(1+x_min**2) \
            + 2*atan((x_max-x_min) / (1 + x_min*x_max))
        v = (4/(x_max-x_min)) * (atan( (x_max-x_min)/(1+x_min*x_max) ))**2
        s_fc = sqrt( pi / (u - v) )
        sigma_fc = fc * s_fc / sqrt( pi * fc * ps.T_measure )

        s_D = sqrt( u / ( (1+pi/2)*(x_max-x_min) ) ) * s_fc
        sigma_D = D * sqrt( (1 + pi/2) / (pi * fc * ps.T_measure) ) * s_D

        return FitResults(fc, D, sigma_fc, sigma_D, ps_fit)

    except ValueError:
        return None


def _alpha(a):
    return 1/sqrt(1+a**2)


def _a(alpha):
    return sqrt(1/alpha**2 - 1)


class FullPSFitModel:
    """Callable wrapper around our model function for the full power spectrum.

    Takes care of fit parameter rescaling, before calling into the core model
    function `P`.
    """

    def __init__(self, scale_factors):
        self.scale_factors = scale_factors

    def __call__(self, f, p1, p2, p3, p4):
        """This method gets called when we try to call a FullPSFitModel object"""
        return self.P(f, *self.get_params_from_rescaled_params((p1, p2, p3, p4)))

    @staticmethod
    def P(f, fc, D, f_diode, alpha):
        """Theoretical model for the full power spectrum.

        See ref. 1, Eq. (10), and ref. 2, Eq. (11).

        Parameters
        ----------
        f : numpy.ndarray
            Frequency values, in Hz.
        fc : float
            Corner frequency, in Hz.
        D : float
            Diffusion constant, in (a.u.)^2/s
        f_diode : float
            Diode fall-off frequency, in Hz.
        alpha : float
            Diode parameter, between 0 and 1.
        """
        return (D/(2*pi**2)) / (f**2 + fc**2) * FullPSFitModel.g_diode(f, f_diode, alpha)

    @staticmethod
    def g_diode(f, f_diode, alpha):
        """Theoretical model for the low-pass filtering by the PSD.

        See ref. 2, Eq. (11).
        """
        return alpha**2 + (1-alpha**2) / (1 + (f/f_diode)**2)

    def get_params_from_rescaled_params(self, rescaled_params):
        return (
            rescaled_params[0] * self.scale_factors[0],
            rescaled_params[1] * self.scale_factors[1],
            rescaled_params[2] * self.scale_factors[2],
            _alpha(rescaled_params[3] * self.scale_factors[3])
        )


def sphere_friction_coefficient(eta, d):
    """Friction coefficient of a sphere with diameter `d` in a liquid with viscosity `eta`

    Parameters
    ----------
    eta : float
        Dynamic / shear viscosity [Pa*s]
    d : float
        Sphere diameter [m]
    """
    return 3*pi*eta*d


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
        self.bead_diameter  = bead_diameter
        self.viscosity      = 1.002e-3
        self.temperature    = 20

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise TypeError('Unknown argument %s' % k)

    def temperature_K(self):
        return scipy.constants.convert_temperature(self.temperature, 'C', 'K')


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
        self.n_points_per_block         = 350
        self.fit_range                  = (1e2, 23e3)
        self.analytical_fit_range       = (1e1, 1e4)
        self.ftol                       = 1e-7
        self.maxfev                     = 10000

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise TypeError('Unknown argument %s' % k)


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
        _valid_attr = ['fc', 'D', 'f_diode', 'alpha',
                       'err_fc', 'err_D', 'err_f_diode', 'err_alpha',
                       'chi_squared_per_deg', 'backing',
                       'ps_fitted', 'ps_model_fit',
                       'Rd', 'kappa', 'Rf', 'error']

        for k, v in kwargs.items():
            if k in _valid_attr:
                setattr(self, k, v)
            else:
                raise TypeError('Unknown argument/attribute %s' % k)

    def is_success(self):
        return not hasattr(self, 'error')


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

        See Also
        --------
        fit_full_powerspectrum
        """
        f_nyquist = ps.sampling_rate/2
        P_aliased_nyq = (guess_D/(2*pi**2)) / (f_nyquist**2 + guess_fc**2)
        if ps.P[-1] < P_aliased_nyq:
            dif = ps.P[-1] / P_aliased_nyq
            return sqrt(dif * f_nyquist**2 / (1. - dif))
        else:
            return 2*f_nyquist

    def run_fit(self, print_diagnostics=False):
        """Runs the actual fitting procedure

        Parameters
        ----------
        print_diagnostics : bool
            If True, prints diagnostics about the fitting procedure to STDOUT.
        """
        # Filter and block the power spectrum.
        ps = self.ps.in_range(*self.settings.fit_range)
        ps = ps.block_averaged(n_blocks=ps.P.size//self.settings.n_points_per_block)

        try:
            # First do an analytical simple Lorentzian fit, to get some initial
            # parameter guesses for the fit.
            anl_fit_ps = ps.in_range(*self.settings.analytical_fit_range)
            anl_fit_res = fit_analytical_lorentzian(anl_fit_ps)
            if not anl_fit_res:
                raise ValueError('Analytical fit failed')
            initial_params = (anl_fit_res.fc, anl_fit_res.D,
                              self.guess_f_diode_initial_value(ps, anl_fit_res.fc, anl_fit_res.D),
                              0.3)

            if print_diagnostics:
                print('Initial fit parameters:   fc = %.2e  D = %.2f  f_diode = %.2e  alpha = %.2f' % initial_params)

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

            model = FullPSFitModel(scale_factors=(*initial_params[0:3], _a(initial_params[3]),))
            sigma = (1/ps.P) / sqrt(self.settings.n_points_per_block)
            (solution_params_rescaled, pcov) = \
                    scipy.optimize.curve_fit(
                        lambda f, fc, D, f_diode, a: 1/model(f, fc, D, f_diode, a),
                        ps.f, 1/ps.P,
                        p0=np.ones(4),
                        sigma=sigma,
                        absolute_sigma=True,
                        method='lm',
                        ftol=self.settings.ftol,
                        maxfev=self.settings.maxfev,
                    )
            solution_params_rescaled = np.abs(solution_params_rescaled)
            # the model function is symmetric in alpha and f_diode...
            solution_params = model.get_params_from_rescaled_params(solution_params_rescaled)

            # Calculate goodness-of-fit, in terms of the statistical backing (see ref. 1).
            chi_squared = np.sum( ( (1/model(ps.f, *solution_params_rescaled) - 1/ps.P) / sigma)**2 )
            n_degrees_of_freedom = ps.P.size - len(solution_params)
            chi_squared_per_deg = chi_squared / n_degrees_of_freedom
            backing = (1 - scipy.special.gammainc(chi_squared/2, n_degrees_of_freedom/2)) * 100

            # We also have to un-rescale the covariance matrix.
            # There's actually a rescaling factor *squared* in there, as the Jacobian
            # appears twice in the equation for the covariance matrix.
            perr = np.sqrt(np.diag(pcov) * (np.array(model.scale_factors)**2))

            # TODO Fix calculation of alpha confidence interval.
            # The previous step calculated the confidence interval in the transformed
            # variable 'a', *not* in 'alpha'! We're using this rather ugly, most likely
            # not-quite-statistically-correct trick to transform the 'a' confidence
            # interval into an 'alpha' confidence interval. Note that this also seems
            # to give us different results for the alpha confidence interval than the
            # original tweezercalib-2.1 code from ref. 3.
            perr[3] = abs(  _alpha(solution_params_rescaled[3]*model.scale_factors[3] + perr[3])
                          - _alpha(solution_params_rescaled[3]*model.scale_factors[3] - perr[3]))/2

            # Fitted power spectrum values.
            ps_model_fit = PowerSpectrum()
            ps_model_fit.f = ps.f
            ps_model_fit.P = model.P(ps.f, *solution_params)
            ps_model_fit.sampling_rate = ps.sampling_rate
            ps_model_fit.T_measure = ps.T_measure

            if print_diagnostics:
                print('Solution:   fc = %.2e  D = %.2f  f_diode = %.2e  alpha = %.2f' % solution_params)
                print('Errors:     fc = %.2e  D = %.2f  f_diode = %.2e  alpha = %.2f' % tuple(perr))
                print('Units:      fc : Hz        D : V^2/s f_diode : Hz')
                print('Chi^2 per degree of freedom = %.2f' % chi_squared_per_deg)
                print('Statistical backing = %.1f%%' % backing)

            # Calculate additional calibration constants.
            gamma_0 = sphere_friction_coefficient(self.params.viscosity, self.params.bead_diameter*1e-6)
            Rd = sqrt(scipy.constants.k * self.params.temperature_K() / gamma_0 / solution_params[1]) * 1e6
            kappa = 2 * pi * gamma_0 * solution_params[0] * 1e3
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
                                Rf=Rf
            )
        except (ValueError, RuntimeError) as e:
            self.results = CalibrationResults(
                                ps_fitted=ps)
            raise CalibrationError(str(e))
