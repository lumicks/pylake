import math
import scipy
from collections import namedtuple
from .detail.power_models import passive_power_spectrum_model, sphere_friction_coefficient


class CalibrationModel:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

    def calibration_parameters(self, params):
        raise NotImplementedError


class PassiveCalibrationModel(CalibrationModel):
    """Model to fit data acquired during passive calibration.

    The power spectrum calibration algorithm implemented here is based on a number of publications
    by the Flyvbjerg group at DTU [1]_ [2]_ [3]_ [4]_.

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

    Attributes
    ----------
    bead_diameter : float
        Bead diameter [um].
    viscosity : float, optional
        Liquid viscosity [Pa*s]. Default: 1.002e-3 Pa*s.
    temperature : float, optional
        Liquid temperature [Celsius].
    """
    def __init__(self, bead_diameter, viscosity=1.002e-3, temperature=20):
        self.bead_diameter = bead_diameter
        self.viscosity = viscosity
        self.temperature = temperature

    def __call__(self, f, fc, diffusion_constant, f_diode, alpha):
        return passive_power_spectrum_model(f, fc, diffusion_constant, f_diode, alpha)

    def calibration_parameters(self, fc, diffusion_constant):
        """Compute calibration parameters from cutoff frequency and diffusion constant.

        Parameters
        ----------
        fc : float
            Corner frequency, in Hz.
        diffusion_constant : float
            Diffusion constant, in (a.u.)^2/s

        Returns
        -------
        namedtuple (Rd, kappa, Rf)
            Attributes:
                Rd : float
                    Distance response [um/V]
                kappa : float
                    Trap stiffness [pN/nm]
                Rf : float
                    Force response [pN/V]
            Note: returns None if the fit fails.
        """
        CalibrationParameters = namedtuple("PassiveCalibrationFitResults", ["Rd", "kappa", "Rf"])

        gamma_0 = sphere_friction_coefficient(self.viscosity, self.bead_diameter * 1e-6)
        temperature_k = scipy.constants.convert_temperature(self.temperature, "C", "K")
        Rd = math.sqrt(scipy.constants.k * temperature_k / gamma_0 / diffusion_constant) * 1e6
        kappa = 2 * math.pi * gamma_0 * fc * 1e3
        Rf = Rd * kappa * 1e3

        return CalibrationParameters(Rd, kappa, Rf)
