import math
import scipy
from .detail.power_models import passive_power_spectrum_model, sphere_friction_coefficient
from .power_spectrum_calibration import CalibrationParameter


class CalibrationModel:
    def __init__(self, bead_diameter):
        self.bead_diameter = bead_diameter

    def __name__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__name__()}({''.join([f'{k}={v}, ' for k, v in vars(self).items()])[:-2]})"

    def calibration_parameters(self, fc, diffusion_constant):
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

    def __name__(self):
        return "PassiveCalibrationModel"

    def __init__(self, bead_diameter, viscosity=1.002e-3, temperature=20):
        super().__init__(bead_diameter)
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
        """
        # diameter [um] -> [m]
        gamma_0 = sphere_friction_coefficient(self.viscosity, self.bead_diameter * 1e-6)
        temperature_k = scipy.constants.convert_temperature(self.temperature, "C", "K")

        # Rd needs to be output in um/V (m -> um or 1e6)
        Rd = math.sqrt(scipy.constants.k * temperature_k / gamma_0 / diffusion_constant) * 1e6

        # Kappa is output in pN/nm (N/m -> pN/nm or 1e12 / 1e9 = 1e3)
        kappa = 2 * math.pi * gamma_0 * fc * 1e3

        # Rf is output in pN/V. Rd [um/V], kappa [pN/nm]: um -> nm = 1e3
        Rf = Rd * kappa * 1e3

        return {
            "Bead diameter (um)": CalibrationParameter("Bead diameter", self.bead_diameter, "um"),
            "Viscosity (Pa*s)": CalibrationParameter("Liquid viscosity", self.viscosity, "Pa*s"),
            "Temperature (C)": CalibrationParameter("Liquid temperature", self.temperature, "C"),
            "Rd (um/V)": CalibrationParameter("Distance response", Rd, "um/V"),
            "kappa (pN/nm)": CalibrationParameter("Trap stiffness", kappa, "pN/nm"),
            "Rf (pN/V)": CalibrationParameter("Force response", Rf, "pN/V"),
        }
