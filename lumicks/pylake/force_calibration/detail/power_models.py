import math
import numpy as np
from collections import namedtuple
from .power_spectrum import PowerSpectrum


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
    FitResults = namedtuple("AnalyticalLorentzianFitResults", ["fc", "D", "sigma_fc", "sigma_D", "ps_fit"])

    # Calculate S[p,q] elements (Ref. 1, Eq. 13-14).
    Spq = np.zeros((3, 3))
    for p in range(3):
        for q in range(3):
            Spq[p, q] = np.sum(np.power(ps.f, 2 * p) * np.power(ps.P, q))
    try:
        # Calculate a and b parameters (Ref. 1, Eq. 13-14).
        a = (Spq[0, 1] * Spq[2, 2] - Spq[1, 1] * Spq[1, 2]) / (Spq[0, 2] * Spq[2, 2] - Spq[1, 2] * Spq[1, 2])
        b = (Spq[1, 1] * Spq[0, 2] - Spq[0, 1] * Spq[1, 2]) / (Spq[0, 2] * Spq[2, 2] - Spq[1, 2] * Spq[1, 2])

        # Having a and b, calculating fc and D is trivial.
        fc = math.sqrt(a / b)  # corner frequency [Hz]
        D = (1 / b) * 2 * (math.pi ** 2)  # diffusion constant [V^2/s]

        # Fitted power spectrum values.
        ps_fit = PowerSpectrum()
        ps_fit.f = ps.f
        ps_fit.P = 1 / (a + b * np.power(ps.f, 2))
        ps_fit.sampling_rate = ps.sampling_rate
        ps_fit.T_measure = ps.T_measure

        # Error propagation (Ref. 1, Eq. 25-28).
        x_min = ps.f.min() / fc
        x_max = ps.f.max() / fc

        u = (
            (2 * x_max) / (1 + x_max ** 2)
            - (2 * x_min) / (1 + x_min ** 2)
            + 2 * math.atan((x_max - x_min) / (1 + x_min * x_max))
        )
        v = (4 / (x_max - x_min)) * (math.atan((x_max - x_min) / (1 + x_min * x_max))) ** 2
        s_fc = math.sqrt(math.pi / (u - v))
        sigma_fc = fc * s_fc / math.sqrt(math.pi * fc * ps.T_measure)

        s_D = math.sqrt(u / ((1 + math.pi / 2) * (x_max - x_min))) * s_fc
        sigma_D = D * math.sqrt((1 + math.pi / 2) / (math.pi * fc * ps.T_measure)) * s_D

        return FitResults(fc, D, sigma_fc, sigma_D, ps_fit)
    except ValueError:
        return None


def _alpha(a):
    return 1 / math.sqrt(1 + a ** 2)


def _a(alpha):
    return math.sqrt(1 / alpha ** 2 - 1)


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
        return (D / (2 * math.pi ** 2)) / (f ** 2 + fc ** 2) * FullPSFitModel.g_diode(f, f_diode, alpha)

    @staticmethod
    def g_diode(f, f_diode, alpha):
        """Theoretical model for the low-pass filtering by the PSD.

        See ref. 2, Eq. (11).
        """
        return alpha ** 2 + (1 - alpha ** 2) / (1 + (f / f_diode) ** 2)

    def get_params_from_rescaled_params(self, rescaled_params):
        return (
            rescaled_params[0] * self.scale_factors[0],
            rescaled_params[1] * self.scale_factors[1],
            rescaled_params[2] * self.scale_factors[2],
            _alpha(rescaled_params[3] * self.scale_factors[3]),
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
    return 3.0 * math.pi * eta * d
