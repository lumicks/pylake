import math
import numpy as np
from collections import namedtuple
from itertools import product
from lumicks.pylake.force_calibration.power_spectrum import PowerSpectrum


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
        - `ps_fit`: `PowerSpectrum` object with model fit

        Note: returns None if the fit fails.
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
    fc = math.sqrt(a / b)  # corner frequency [Hz]
    D = (1 / b) * (math.pi ** 2)  # diffusion constant [V^2/s]

    # Fitted power spectrum values.
    ps_fit = ps.with_spectrum(1 / (a + b * np.power(ps.frequency, 2)))

    # Error propagation (Ref. 1, Eq. 25-28).
    x_min = ps.frequency.min() / fc
    x_max = ps.frequency.max() / fc

    u = (
        (2 * x_max) / (1 + x_max ** 2)
        - (2 * x_min) / (1 + x_min ** 2)
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
    return alpha ** 2 + (1 - alpha ** 2) / (1 + (f / f_diode) ** 2)


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
    return (diffusion_constant / (math.pi ** 2)) / (f ** 2 + fc ** 2)


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

    fc : float
        Corner frequency [Hz]
    driving_frequency : float
        Driving frequency [Hz]
    driving_amplitude : float
        Driving amplitude [m]
    """
    return driving_amplitude ** 2 / (2 * (1 + (fc / driving_frequency) ** 2))
