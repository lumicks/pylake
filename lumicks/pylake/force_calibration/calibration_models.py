import numpy as np
import scipy
from .detail.driving_input import (
    estimate_driving_input_parameters,
    driving_power_peak,
)
from .detail.power_models import passive_power_spectrum_model, sphere_friction_coefficient
from .power_spectrum_calibration import CalibrationParameter


class CalibrationModel:
    def __init__(self, bead_diameter):
        self.bead_diameter = bead_diameter

    def __name__(self):
        raise NotImplementedError

    def __call__(self, f, fc, diffusion_constant, f_diode, alpha):
        return passive_power_spectrum_model(f, fc, diffusion_constant, f_diode, alpha)

    def __repr__(self):
        return f"{self.__name__()}({''.join([f'{k}={v}, ' for k, v in vars(self).items()])[:-2]})"

    def calibration_parameters(self):
        raise NotImplementedError

    def calibration_results(self, fc, diffusion_constant_volts, f_diode, alpha):
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
        Liquid viscosity [Pa*s].
    temperature : float, optional
        Liquid temperature [Celsius].
    """

    def __name__(self):
        return "PassiveCalibrationModel"

    def __init__(self, bead_diameter, viscosity=1.002e-3, temperature=20):
        super().__init__(bead_diameter)
        self.viscosity = viscosity
        self.temperature = temperature

    def calibration_parameters(self):
        return {
            "Bead diameter": CalibrationParameter("Bead diameter", self.bead_diameter, "um"),
            "Viscosity": CalibrationParameter("Liquid viscosity", self.viscosity, "Pa*s"),
            "Temperature": CalibrationParameter("Liquid temperature", self.temperature, "C"),
        }

    def calibration_results(self, fc, diffusion_constant_volts, f_diode, alpha):
        """Compute calibration parameters from cutoff frequency and diffusion constant.

        Note: f_diode and alpha are not used for passive calibration and are there to maintain
        the same call signature for active and passive calibration.

        Parameters
        ----------
        fc : float
            Corner frequency, in Hz.
        diffusion_constant_volts : float
            Diffusion constant, in V^2/s
        f_diode : float
            Diode frequency.
        alpha : float
            Fraction of PSD signal that is instantaneous
        """
        # diameter [um] -> [m]
        gamma_0 = sphere_friction_coefficient(self.viscosity, self.bead_diameter * 1e-6)
        temperature_k = scipy.constants.convert_temperature(self.temperature, "C", "K")

        # Distance response (Rd) needs to be output in um/V (m -> um or 1e6)
        distance_response = (
            np.sqrt(scipy.constants.k * temperature_k / gamma_0 / diffusion_constant_volts) * 1e6
        )

        # Stiffness is output in pN/nm (N/m -> pN/nm or 1e12 / 1e9 = 1e3)
        kappa = 2 * np.pi * gamma_0 * fc * 1e3

        # Force response (Rf) is output in pN/V. Rd [um/V], stiffness [pN/nm]: um -> nm = 1e3
        force_response = distance_response * kappa * 1e3

        return {
            "Rd": CalibrationParameter("Distance response", distance_response, "um/V"),
            "kappa": CalibrationParameter("Trap stiffness", kappa, "pN/nm"),
            "Rf": CalibrationParameter("Force response", force_response, "pN/V"),
        }


class ActiveCalibrationModel(CalibrationModel):
    """Model to fit data acquired during active calibration.

    The power spectrum calibration algorithm implemented here is based on [1]_, [2]_, [3]_, [4]_,
    [5]_, [6]_.

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

    Attributes
    ----------
    sample_rate : float
        Sample rate at which the signals we acquired.
    bead_diameter : float
        Bead diameter [um].
    driving_frequency : float
        Estimated driving frequency [Hz].
    driving_amplitude : float
        Estimated driving amplitude [m].
    viscosity : float, optional
        Liquid viscosity [Pa*s].
    temperature : float, optional
        Liquid temperature [Celsius].
    num_windows : int, optional
        Number of windows to average for the uncalibrated force. Using a larger number of
        windows potentially increases the bleed, but may be useful when the SNR is low.
    """

    def __name__(self):
        return "ActiveCalibrationModel"

    def __init__(
        self,
        driving_data,
        force_voltage_data,
        sample_rate,
        bead_diameter,
        driving_frequency_guess,
        viscosity=1.002e-3,
        temperature=20,
        num_windows=5,
    ):
        """
        Active Calibration Model.

        This model fits data acquired in the presence of nanostage or mirror oscillation.

        Parameters
        ----------
        driving_data : array_like
            Array of driving data.
        force_voltage_data : array_like
            Uncalibrated force data in volts.
        sample_rate : float
            Sample rate at which the signals we acquired.
        bead_diameter : float
            Bead diameter [um].
        driving_frequency_guess : float
            Guess of the driving frequency.
        viscosity : float, optional
            Liquid viscosity [Pa*s].
        temperature : float, optional
            Liquid temperature [Celsius].
        num_windows : int, optional
            Number of windows to average for the uncalibrated force. Using a larger number of
            windows potentially increases the bleed, but may be useful when the SNR is low.
        """
        super().__init__(bead_diameter)
        self.driving_frequency_guess = driving_frequency_guess
        self.sample_rate = sample_rate
        self.viscosity = viscosity
        self.temperature = temperature
        self.num_windows = num_windows

        # Estimate driving input and response
        amplitude_um, self.driving_frequency = estimate_driving_input_parameters(
            sample_rate, driving_data, driving_frequency_guess
        )
        self.driving_amplitude = amplitude_um * 1e-6

        # The power density is the density (V^2/Hz) at the driving input. Multiplying this with
        # the frequency bin width gives the absolute power.
        self._response_power_density, self._frequency_bin_width = driving_power_peak(
            force_voltage_data, sample_rate, self.driving_frequency, num_windows
        )

    def calibration_parameters(self):
        return {
            "Bead diameter": CalibrationParameter("Bead diameter", self.bead_diameter, "um"),
            "Driving frequency (guess)": CalibrationParameter(
                "Driving frequency (guess)", self.driving_frequency_guess, "Hz"
            ),
            "Sample rate": CalibrationParameter("Sample rate", self.sample_rate, "Hz"),
            "Temperature": CalibrationParameter("Liquid temperature", self.temperature, "C"),
            "Viscosity": CalibrationParameter("Liquid viscosity", self.viscosity, "Pa*s"),
            "num_windows": CalibrationParameter("Number of averaged windows", self.num_windows, ""),
        }

    def _theoretical_driving_power(self, f_corner):
        """Compute the power expected for a given driving input.

        When driving the stage or trap, we expect to see a delta spike in the power density
        spectrum. This function returns the expected power contribution of the bead motion to the
        power spectrum. It corresponds to the driven power spectrum minus the thermal power spectrum
        integrated over the frequency bin corresponding to the driving input."""
        return self.driving_amplitude ** 2 / (2 * (1 + (f_corner / self.driving_frequency) ** 2))

    def calibration_results(self, fc, diffusion_constant_volts, f_diode, alpha):
        """Compute calibration parameters from cutoff frequency and diffusion constant.

        Parameters
        ----------
        fc : float
            Corner frequency, in Hz.
        diffusion_constant_volts : float
            Diffusion constant, in V^2/s
        f_diode : float
            Diode frequency.
        alpha : float
            Fraction of PSD signal that is instantaneous
        """
        reference_peak = self(self.driving_frequency, fc, diffusion_constant_volts, f_diode, alpha)

        power_exp = (self._response_power_density - reference_peak) * self._frequency_bin_width
        distance_response = np.sqrt(self._theoretical_driving_power(fc) / power_exp)  # m/V

        temperature_kelvin = scipy.constants.convert_temperature(self.temperature, "C", "K")
        k_temperature = scipy.constants.k * temperature_kelvin
        gamma_0 = k_temperature / (distance_response ** 2 * diffusion_constant_volts)  # kg/s^2
        kappa = 2.0 * np.pi * fc * gamma_0  # N/m

        force_response = distance_response * kappa  # N/V

        return {
            "Rd": CalibrationParameter("Distance response", distance_response * 1e6, "um/V"),
            "kappa": CalibrationParameter("Trap stiffness", kappa * 1e3, "pN/nm"),
            "Rf": CalibrationParameter("Force response", force_response * 1e12, "pN/V"),
            "gamma_0": CalibrationParameter("Drag coefficient", gamma_0, "kg/s**2"),
        }
