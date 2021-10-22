import numpy as np
import scipy
from functools import partial
from .detail.power_models import g_diode
from .detail.driving_input import (
    estimate_driving_input_parameters,
    driving_power_peak,
)
from .detail.power_models import (
    passive_power_spectrum_model,
    sphere_friction_coefficient,
    theoretical_driving_power_lorentzian,
)
from .detail.hydrodynamics import (
    passive_power_spectrum_model_hydro,
    theoretical_driving_power_hydrodynamics,
)
from .power_spectrum_calibration import CalibrationParameter

from dataclasses import dataclass
from typing import Callable


@dataclass
class Param:
    name: str
    description: str
    unit: str
    initial: float
    lower_bound: float
    upper_bound: Callable[[float], float]


class FilterBase:
    def __init__(self):
        self.fitted_params = []

    def lower_bounds(self):
        return [p.lower_bound for p in self.fitted_params]

    def upper_bounds(self, sample_rate):
        return [p.upper_bound(sample_rate) for p in self.fitted_params]

    @property
    def initial_values(self):
        return [p.initial for p in self.fitted_params]

    def params(self):
        """Parameters that were fixed during the fitting"""
        return {}

    def results(self, values, std_errs):
        """Fitted parameter values"""
        return {
            **{
                p.name: CalibrationParameter(p.description, val, p.unit)
                for val, p in zip(values, self.fitted_params)
            },
            **{
                f"err_{p.name}": CalibrationParameter(p.description + " Std Err", err, p.unit)
                for err, p in zip(std_errs, self.fitted_params)
            },
        }


class NoFilter(FilterBase):
    def __call__(self, *_):
        return 1


class DiodeModel(FilterBase):
    def __init__(self):
        self.fitted_params = [
            Param(
                name="f_diode",
                description="Diode low-pass filtering roll-off frequency",
                unit="Hz",
                initial=14000,
                lower_bound=0.0,
                upper_bound=lambda sample_rate: sample_rate / 2,
            ),
            Param(
                name="alpha",
                description="Diode 'relaxation factor'",
                unit="",
                initial=0.3,
                lower_bound=0.0,
                upper_bound=lambda _: 1.0,
            ),
        ]

    def __call__(self, f, *pars):
        return g_diode(f, *pars)


class FixedDiodeModel(FilterBase):
    """Model with fixed diode frequency"""

    def __init__(self, diode_frequency):
        self.diode_frequency = diode_frequency
        self.fitted_params = [
            Param(
                name="alpha",
                description="Diode 'relaxation factor'",
                unit="",
                initial=0.3,
                lower_bound=0.0,
                upper_bound=lambda _: 1.0,
            )
        ]

    def params(self):
        """Parameters that were fixed during the fitting"""
        return {
            "f_diode": CalibrationParameter(
                "Diode low-pass filtering roll-off frequency (fixed)", self.diode_frequency, "Hz"
            )
        }

    def __call__(self, f, alpha):
        return g_diode(f, self.diode_frequency, alpha)


class PassiveCalibrationModel:
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
    .. [5] Tolić-Nørrelykke, S. F, and Flyvbjerg, H, "Power spectrum analysis with least-squares
           fitting: amplitude bias and its elimination, with application to optical tweezers and
           atomic force microscope cantilevers." Review of Scientific Instruments 81.7 (2010)
    .. [6] Tolić-Nørrelykke S. F, Schäffer E, Howard J, Pavone F. S, Jülicher F and Flyvbjerg, H.
           Calibration of optical tweezers with positional detection in the back focal plane,
           Review of scientific instruments 77, 103101 (2006).

    Attributes
    ----------
    bead_diameter : float
        Bead diameter [um].
    viscosity : float, optional
        Liquid viscosity [Pa*s].
    temperature : float, optional
        Liquid temperature [Celsius].
    hydrodynamically_correct : bool, optional
        Enable hydrodynamic correction.
    distance_to_surface : float, optional
        Distance from bead center to the surface [um].
        Currently not supported for models without hydrodynamics.
    rho_sample : float, optional
        Density of the sample [kg/m^3]. Only used when using hydrodynamic corrections.
    rho_bead : float, optional
        Density of the bead [kg/m^3]. Only used when using hydrodynamic corrections.
    fast_sensor : bool
        Fast sensor? Fast sensors do not have the diode effect included in the model.
    """

    def __name__(self):
        return "PassiveCalibrationModel"

    def __init__(
        self,
        bead_diameter,
        viscosity=1.002e-3,
        temperature=20,
        hydrodynamically_correct=False,
        distance_to_surface=None,
        rho_sample=None,
        rho_bead=1060.0,
        fast_sensor=False,
    ):
        if bead_diameter < 1e-2:
            raise ValueError(
                f"Invalid bead diameter specified {bead_diameter}. Bead diameter should be bigger "
                f"than 10^-2 um"
            )

        if (
            distance_to_surface
            and hydrodynamically_correct
            and distance_to_surface < bead_diameter / 2.0
        ):
            raise ValueError("Distance from bead center to surface is smaller than the bead radius")

        self.viscosity = viscosity
        self.temperature = temperature
        self.bead_diameter = bead_diameter
        self.drag_coeff = sphere_friction_coefficient(self.viscosity, self.bead_diameter * 1e-6)
        self._filter = NoFilter() if fast_sensor else DiodeModel()

        self.hydrodynamically_correct = hydrodynamically_correct
        # Note that the default is set to None because in the future we may want to provide an
        # estimate of rho_sample based on temperature. If we pin this to a value already, this
        # would become a breaking change.
        self.rho_sample = rho_sample if rho_sample is not None else 997.0
        self.rho_bead = rho_bead
        self.distance_to_surface = distance_to_surface

        if hydrodynamically_correct:
            # This model is only valid up to l/R < 1.5 [6] so throw in case that is violated.
            if distance_to_surface and distance_to_surface / (self.bead_diameter / 2) < 1.5:
                raise ValueError(
                    "This model is only valid for distances to the surface larger "
                    "than 1.5 times the bead radius. Distances closer to the surface "
                    "are currently not supported."
                )

            self._passive_power_spectrum_model = partial(
                passive_power_spectrum_model_hydro,
                gamma0=self.drag_coeff,
                bead_radius=self.bead_diameter * 1e-6 / 2.0,  # um diameter -> m radius
                rho_sample=self.rho_sample,
                rho_bead=self.rho_bead,
                distance_to_surface=None
                if self.distance_to_surface is None
                else self.distance_to_surface * 1e-6,  # um => m
            )
        else:
            if self.distance_to_surface:
                raise NotImplementedError(
                    "Using a distance to the surface is currently not supported for models without "
                    "hydrodynamic effects."
                )

            self._passive_power_spectrum_model = passive_power_spectrum_model

    def __call__(self, f, fc, diffusion_constant, *filter_params):
        physical_spectrum = self._passive_power_spectrum_model(f, fc, diffusion_constant)
        return physical_spectrum * self._filter(f, *filter_params)

    def calibration_parameters(self):
        hydrodynamic_parameters = (
            {
                "Sample density": CalibrationParameter(
                    "Density of the sample", self.rho_sample, "kg/m**3"
                ),
                "Bead density": CalibrationParameter(
                    "Density of bead material", self.rho_bead, "kg/m**3"
                ),
                "Distance to surface": CalibrationParameter(
                    "Distance from bead center to surface", self.distance_to_surface, "um"
                ),
            }
            if self.hydrodynamically_correct
            else {}
        )

        return {
            "Bead diameter": CalibrationParameter("Bead diameter", self.bead_diameter, "um"),
            "Viscosity": CalibrationParameter("Liquid viscosity", self.viscosity, "Pa*s"),
            "Temperature": CalibrationParameter("Liquid temperature", self.temperature, "C"),
            **self._filter.params(),
            **hydrodynamic_parameters,
        }

    def _format_passive_result(
        self,
        fc,
        diffusion_constant_volts,
        filter_params,
        fc_err,
        diffusion_constant_volts_err,
        filter_params_err,
    ):
        """Format the fit parameters"""
        return {
            "fc": CalibrationParameter("Corner frequency", fc, "Hz"),
            "D": CalibrationParameter("Diffusion constant", diffusion_constant_volts, "V^2/s"),
            "err_fc": CalibrationParameter("Corner frequency Std Err", fc_err, "Hz"),
            "err_D": CalibrationParameter(
                "Diffusion constant Std Err", diffusion_constant_volts_err, "V^2/s"
            ),
            **self._filter.results(filter_params, filter_params_err),
        }

    def calibration_results(
        self,
        fc,
        diffusion_constant_volts,
        filter_params,
        fc_err,
        diffusion_constant_volts_err,
        filter_params_err,
    ):
        """Compute calibration parameters from cutoff frequency and diffusion constant.

        Parameters
        ----------
        fc : float
            Corner frequency, in Hz.
        diffusion_constant_volts : float
            Diffusion constant, in V^2/s
        filter_params : list of float
            Parameters for the filter model.
        fc_err : float
            Corner frequency standard error, in Hz
        diffusion_constant_volts_err : float
            Diffusion constant standard error, in Hz
        filter_params_err : list of float
            Standard errors for the filter model
        """
        # diameter [um] -> [m]
        temperature_k = scipy.constants.convert_temperature(self.temperature, "C", "K")

        # Distance response (Rd) needs to be output in um/V (m -> um or 1e6)
        distance_response = (
            np.sqrt(scipy.constants.k * temperature_k / self.drag_coeff / diffusion_constant_volts)
            * 1e6
        )

        # Stiffness is output in pN/nm (N/m -> pN/nm or 1e12 / 1e9 = 1e3)
        kappa = 2 * np.pi * self.drag_coeff * fc * 1e3

        # Force response (Rf) is output in pN/V. Rd [um/V], stiffness [pN/nm]: um -> nm = 1e3
        force_response = distance_response * kappa * 1e3

        return {
            "Rd": CalibrationParameter("Distance response", distance_response, "um/V"),
            "kappa": CalibrationParameter("Trap stiffness", kappa, "pN/nm"),
            "Rf": CalibrationParameter("Force response", force_response, "pN/V"),
            "gamma_0": CalibrationParameter(
                "Theoretical drag coefficient", self.drag_coeff, "kg/s"
            ),
            **self._format_passive_result(
                fc,
                diffusion_constant_volts,
                filter_params,
                fc_err,
                diffusion_constant_volts_err,
                filter_params_err,
            ),
        }


class ActiveCalibrationModel(PassiveCalibrationModel):
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
    hydrodynamically_correct : bool, optional
        Enable hydrodynamically correct spectrum.
    distance_to_surface : float, optional
        Distance from bead center to the surface [um].
        Currently not supported for models without hydrodynamics.
    rho_sample : float, optional
        Density of the sample [kg/m^3]. Only used when using hydrodynamic corrections.
    rho_bead : float, optional
        Density of the bead [kg/m^3]. Only used when using hydrodynamic corrections.
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
        hydrodynamically_correct=False,
        distance_to_surface=None,
        rho_sample=None,
        rho_bead=1060.0,
        fast_sensor=False,
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
        hydrodynamically_correct : bool, optional
            Enable hydrodynamically correct model.
        distance_to_surface : float, optional
            Distance from bead center to the surface [um]
            Currently not supported for models without hydrodynamics.
        rho_sample : float, optional
            Density of the sample [kg/m**3]. Only used when using hydrodynamically correct model.
        rho_bead : float, optional
            Density of the bead [kg/m**3]. Only used when using hydrodynamically correct model.
        fast_sensor : bool
            Fast sensor? Fast sensors do not have the diode effect included in the model.
        """
        super().__init__(
            bead_diameter,
            viscosity,
            temperature,
            hydrodynamically_correct,
            distance_to_surface,
            rho_sample,
            rho_bead,
            fast_sensor,
        )
        self.driving_frequency_guess = driving_frequency_guess
        self.sample_rate = sample_rate
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

        if hydrodynamically_correct:
            self._theoretical_driving_power_model = partial(
                theoretical_driving_power_hydrodynamics,
                driving_frequency=self.driving_frequency,
                driving_amplitude=self.driving_amplitude,
                gamma0=self.drag_coeff,
                bead_radius=self.bead_diameter * 1e-6 / 2.0,  # um diameter -> m radius
                rho_sample=self.rho_sample,
                rho_bead=self.rho_bead,
                distance_to_surface=None
                if self.distance_to_surface is None
                else self.distance_to_surface * 1e-6,  # um => m
            )
        else:
            self._theoretical_driving_power_model = partial(
                theoretical_driving_power_lorentzian,
                driving_frequency=self.driving_frequency,
                driving_amplitude=self.driving_amplitude,
            )

    def calibration_parameters(self):
        return {
            **super().calibration_parameters(),
            "Driving frequency (guess)": CalibrationParameter(
                "Driving frequency (guess)", self.driving_frequency_guess, "Hz"
            ),
            "num_windows": CalibrationParameter("Number of averaged windows", self.num_windows, ""),
        }

    def _theoretical_driving_power(self, f_corner):
        """Compute the power expected for a given driving input.

        When driving the stage or trap, we expect to see a delta spike in the power density
        spectrum. This function returns the expected power contribution of the bead motion to the
        power spectrum. It corresponds to the driven power spectrum minus the thermal power spectrum
        integrated over the frequency bin corresponding to the driving input."""
        return self._theoretical_driving_power_model(f_corner)

    def calibration_results(
        self,
        fc,
        diffusion_constant_volts,
        filter_params,
        fc_err,
        diffusion_constant_volts_err,
        filter_params_err,
    ):
        """Compute active calibration parameters from cutoff frequency and diffusion constant.

        Parameters
        ----------
        fc : float
            Corner frequency, in Hz.
        diffusion_constant_volts : float
            Diffusion constant, in V^2/s
        filter_params : list of float
            Parameters for the filter model.
        fc_err : float
            Corner frequency standard error, in Hz
        diffusion_constant_volts_err : float
            Diffusion constant standard error, in Hz
        filter_params_err : list of float
            Standard errors for the filter model
        """
        reference_peak = self(self.driving_frequency, fc, diffusion_constant_volts, *filter_params)

        # Equation 14 from [6]
        power_exp = (self._response_power_density - reference_peak) * self._frequency_bin_width

        # Equation 12 from [6]
        distance_response = np.sqrt(self._theoretical_driving_power(fc) / power_exp)  # m/V

        # Equation 16 from [6]
        temperature_kelvin = scipy.constants.convert_temperature(self.temperature, "C", "K")
        k_temperature = scipy.constants.k * temperature_kelvin
        measured_drag_coeff = k_temperature / (
            distance_response ** 2 * diffusion_constant_volts
        )  # kg/s

        kappa = 2.0 * np.pi * fc * measured_drag_coeff  # N/m

        force_response = distance_response * kappa  # N/V

        return {
            "Rd": CalibrationParameter("Distance response", distance_response * 1e6, "um/V"),
            "kappa": CalibrationParameter("Trap stiffness", kappa * 1e3, "pN/nm"),
            "Rf": CalibrationParameter("Force response", force_response * 1e12, "pN/V"),
            "gamma_0": CalibrationParameter(
                "Theoretical drag coefficient", self.drag_coeff, "kg/s"
            ),
            "gamma_ex": CalibrationParameter(
                "Measured drag coefficient", measured_drag_coeff, "kg/s"
            ),
            **self._format_passive_result(
                fc,
                diffusion_constant_volts,
                filter_params,
                fc_err,
                diffusion_constant_volts_err,
                filter_params_err,
            ),
        }
