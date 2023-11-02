import math
from itertools import product
from collections import namedtuple

import numpy as np


def fit_analytical_lorentzian(ps):
    """Performs an analytical least-squares fit of a Lorentzian Power Spectrum.

    Based on Section IV from ref. 1. Note that the equations for the statistics Spq are divided
    by a factor of two since we defined the power spectrum in V^2/Hz instead of 0.5 V^2/Hz.

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
        - `ps_fit`: :class:`~lumicks.pylake.force_calibration.power_spectrum.PowerSpectrum` object with model fit

    """
    FitResults = namedtuple(
        "AnalyticalLorentzianFitResults", ["fc", "D", "sigma_fc", "sigma_D", "ps_fit"]
    )

    # Calculate S[p,q] elements (Ref. 1, Eq. 13-14).
    Spq = np.zeros((3, 3))
    for p, q in product(range(3), range(3)):
        Spq[p, q] = np.sum(np.power(ps.frequency, 2 * p) * np.power(ps.power, q))

    # Calculate a and b parameters (Ref. 1, Eq. 13-14).
    a = (Spq[0, 1] * Spq[2, 2] - Spq[1, 1] * Spq[1, 2]) / (
        Spq[0, 2] * Spq[2, 2] - Spq[1, 2] * Spq[1, 2]
    )
    b = (Spq[1, 1] * Spq[0, 2] - Spq[0, 1] * Spq[1, 2]) / (
        Spq[0, 2] * Spq[2, 2] - Spq[1, 2] * Spq[1, 2]
    )

    # Having a and b, calculating fc and D is trivial.
    a_div_b = a / b
    if a_div_b > 0:
        fc = math.sqrt(a_div_b)  # corner frequency [Hz]
    else:
        # When the corner frequency is very low and the power spectrum doesn't reach all the way,
        # this can fail. As initial guess we then use the half the lowest nonzero frequency observed
        # in the power spectrum (optimal when assuming uniform prior for our guess). Note that zero
        # isn't a valid choice, since this leads to nan's and infinities down the road.
        fc = 0.5 * (ps.frequency[0] if ps.frequency[0] > 0 else ps.frequency[1])

    if b > 0:
        D = (1 / b) * (math.pi**2)  # diffusion constant [V^2/s]
    else:
        # If b <= 0, the analytic estimation procedure failed. Using a negative value for the
        # diffusion would place the initial guess outside the feasible physical parameter range.
        # This would result in a failing non-linear fit. In this case, we need a different way of
        # estimating this quantity. The power spectral density at frequency zero is given by:
        #
        #   P0 = D / (pi**2 * fc**2)
        #
        # Therefore, an alternative method to get a rough estimate for this quantity would be:
        #
        #   D_guess = pi**2 * fc**2 * power[0]
        #
        # Where power[0] is the lowest frequency in the spectrum.
        D = math.pi**2 * fc**2 * ps.power[0]

    # Fitted power spectrum values.
    ps_fit = ps.with_spectrum(1 / (a + b * np.power(ps.frequency, 2)))

    # Error propagation (Ref. 1, Eq. 25-28).
    x_min = ps.frequency.min() / fc
    x_max = ps.frequency.max() / fc

    u = (
        (2 * x_max) / (1 + x_max**2)
        - (2 * x_min) / (1 + x_min**2)
        + 2 * math.atan((x_max - x_min) / (1 + x_min * x_max))
    )
    v = (4 / (x_max - x_min)) * (math.atan((x_max - x_min) / (1 + x_min * x_max))) ** 2
    s_fc = math.sqrt(math.pi / (u - v))
    sigma_fc = fc * s_fc / math.sqrt(math.pi * fc * ps.total_duration)

    s_D = math.sqrt(u / ((1 + math.pi / 2) * (x_max - x_min))) * s_fc
    sigma_D = D * math.sqrt((1 + math.pi / 2) / (math.pi * fc * ps.total_duration)) * s_D

    return FitResults(fc, D, sigma_fc, sigma_D, ps_fit)


class ScaledModel:
    """Callable wrapper around a model function to handle scaling"""

    def __init__(self, model, initial_guess):
        self._model = model
        self._scale_factors = initial_guess

    def scale_params(self, rescaled_params):
        return rescaled_params * self._scale_factors

    def normalize_params(self, params):
        return params / self._scale_factors

    def __call__(self, f, *pars):
        return self._model(f, *self.scale_params(pars))


def g_diode(f, f_diode, alpha):
    """Theoretical model for the low-pass filtering by the PSD.

    See ref. 2, Eq. (11).
    """
    return alpha**2 + (1 - alpha**2) / (1 + (f / f_diode) ** 2)


def passive_power_spectrum_model(f, fc, diffusion_constant):
    """Lorentzian model for the power spectrum.

    See ref. 1, Eq. (10), and ref. 2, Eq. (11).
    Note that this implementation deviates from Eq. (10) and Eq. (11) by a factor of 2 since we
    express the power spectrum in V^2 / Hz rather than 0.5 V^2 / Hz.

    Parameters
    ----------
    f : numpy.ndarray
        Frequency values, in Hz.
    fc : float
        Corner frequency, in Hz.
    diffusion_constant : float
        Diffusion constant, in (a.u.)^2/s
    """
    return (diffusion_constant / (math.pi**2)) / (f**2 + fc**2)


def sphere_friction_coefficient(eta, d):
    """Friction coefficient of a sphere with diameter `d` in a liquid with viscosity `eta`

    Parameters
    ----------
    eta : float
        Dynamic / shear viscosity [Pa*s]
    d : float
        Sphere diameter [m]
    """
    return 3.0 * math.pi * eta * d


def theoretical_driving_power_lorentzian(fc, driving_frequency, driving_amplitude):
    """Compute the power expected for a given driving input.

    When driving the stage or trap, we expect to see a delta spike in the power density
    spectrum. This function returns the expected power contribution of the bead motion to the
    power spectrum. It corresponds to the driven power spectrum minus the thermal power spectrum
    integrated over the frequency bin corresponding to the driving input.

    Parameters
    ----------
    fc : float
        Corner frequency [Hz]
    driving_frequency : float
        Driving frequency [Hz]
    driving_amplitude : float
        Driving amplitude [m]
    """
    return driving_amplitude**2 / (2 * (1 + (fc / driving_frequency) ** 2))


def motion_blur_peak(peak, driving_frequency, acquisition_time):
    """Take into account motion blur on the driving peak.

    Parameters
    ----------
    peak : callable
        Function which takes a corner frequency and produces an estimated peak power.
    driving_frequency : float
        Driving frequency.
    acquisition_time : float
        Acquisition time in seconds.

    References
    ----------
    .. [1] Wong, W. P., & Halvorsen, K. (2006). The effect of integration time on fluctuation
       measurements: calibrating an optical trap in the presence of motion blur. Optics express,
       14(25), 12517-12531.
    """

    def blurred(fc, *args, **kwargs):
        return peak(fc, *args, **kwargs) * np.sinc(driving_frequency * acquisition_time) ** 2

    return blurred


def motion_blur_spectrum(psd, acquisition_time):
    """Take into account motion blur on the spectrum.

    Parameters
    ----------
    psd : callable
        Function which takes a numpy array of frequencies and returns a power spectral density.
    acquisition_time : float
        Acquisition time in seconds.

    References
    ----------
    .. [1] Wong, W. P., & Halvorsen, K. (2006). The effect of integration time on fluctuation
       measurements: calibrating an optical trap in the presence of motion blur. Optics express,
       14(25), 12517-12531.
    """

    def blurred(freq, *args, **kwargs):
        return psd(freq, *args, **kwargs) * np.sinc(freq * acquisition_time) ** 2

    return blurred


def alias_spectrum(psd, sample_rate, num_alias=10):
    """
    Produce an aliased version of the input PSD function.

    Parameters
    ----------
    psd : callable
        Function which takes a numpy array of frequencies and returns a power spectral density.
    sample_rate : float
        Sampling frequency.
    num_alias : int
        Number of aliases to simulate. The default of 10 typically gives less than 1% error.
    """

    def aliased(freq, *args, **kwargs):
        """Aliased PSD

        Parameters
        ----------
        freq : numpy.ndarray
            Frequency values.
        *args, **kwargs
            Forwarded to the power spectral density function being wrapped.
        """
        return sum(
            psd(freq + i * sample_rate, *args, **kwargs) for i in range(-num_alias, num_alias + 1)
        )

    return aliased
