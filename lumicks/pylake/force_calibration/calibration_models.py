from copy import copy
from typing import Callable
from functools import partial
from dataclasses import dataclass

import numpy as np

from .detail.drag_models import faxen_factor, brenner_axial
from .detail.power_models import (
    g_diode,
    alias_spectrum,
    sphere_friction_coefficient,
    passive_power_spectrum_model,
    theoretical_driving_power_lorentzian,
)
from .detail.driving_input import driving_power_peak, estimate_driving_input_parameters
from .detail.hydrodynamics import (
    passive_power_spectrum_model_hydro,
    theoretical_driving_power_hydrodynamics,
)
from .power_spectrum_calibration import CalibrationParameter


def diode_params_from_voltage(
    trap_voltage, delta_f_diode, rate_f_diode, max_f_diode, delta_alpha, rate_alpha, max_alpha
):
    """Function to look up the current f_diode and alpha at a given trap power.

    Parameters
    ----------
    trap_voltage : np.array
        Array of trap voltages [V].
    delta_f_diode : float
        Range over which the diode frequency varies [Hz].
    rate_f_diode : float
        Constant which characterizes how quickly the diode frequency changes with power [1/V].
    max_f_diode : float
        Final value for the diode frequency at high powers [Hz].
    delta_alpha : float
        Range over which the diode relaxation factor varies [-].
    rate_alpha : float
        Constant which characterizes how quickly the relaxation factor changes with power [1/V].
    max_alpha : float
        Range over which the diode relaxation factor varies [-].

    Returns
    -------
    f_diode : float
        Diode frequency at this trap power [Hz].
    alpha : float
        Relaxation factor at this trap power [-].
    voltage : float
        Trap voltage [V].
    """
    voltage = np.clip(np.mean(trap_voltage), 0, np.inf)
    return (
        max_f_diode - delta_f_diode * np.exp(-rate_f_diode * voltage),
        max_alpha - delta_alpha * np.exp(-rate_alpha * voltage),
        voltage,
    )


def viscosity_of_water(temperature):
    """Computes the viscosity of water in [Pa*s] at a particular temperature.

    These equations come from section 3.7 from [1]_.

    Parameters
    ----------
    temperature : array_like
        Temperature [Celsius]. Should be between -20°C and 110°C

    Returns
    -------
    array_like
        Viscosity of water [Pa*s]

    Raises
    ------
    ValueError
        If the requested temperature is outside the valid range of -20°C <= T < 110°C.

    References
    ----------
    .. [1] Huber, M. L., Perkins, R. A., Laesecke, A., Friend, D. G., Sengers, J. V.,
           Assael, M. J., & Miyagawa, K. (2009). New international formulation for the viscosity of
           H2O. Journal of Physical and Chemical Reference Data, 38(2), 101-125.
    """
    temperature = np.asarray(temperature)
    if not np.all(np.logical_and(temperature >= -20, temperature < 110)):
        raise ValueError("Function for viscosity of water is only valid for -20°C <= T < 110°C")

    temp_tilde = np.tile((temperature + 273.15) / 300, (4, 1)).T
    ai = np.array([280.68, 511.45, 61.131, 0.45903])
    bi = np.array([-1.9, -7.7, -19.6, -40.0])
    return np.sum(ai * temp_tilde**bi, axis=1).squeeze() * 1e-6


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


# These are used for parameter bounds in the DiodeModel and have to be defined non-locally to ensure
# that the class can be pickled.
def _one(_):
    return 1.0


def _nyquist(sample_rate):
    return sample_rate / 2.0


class DiodeModel(FilterBase):
    def __init__(self):
        self.fitted_params = [
            Param(
                name="f_diode",
                description="Diode low-pass filtering roll-off frequency",
                unit="Hz",
                initial=14000,
                lower_bound=1.0,
                upper_bound=_nyquist,
            ),
            Param(
                name="alpha",
                description="Diode 'relaxation factor'",
                unit="",
                initial=0.3,
                lower_bound=0.0,
                upper_bound=_one,
            ),
        ]

    def __call__(self, f, *pars):
        return g_diode(f, *pars)


class FixedDiodeModel(FilterBase):
    """Model with fixed diode parameters"""

    def __init__(self, diode_frequency=None, diode_alpha=None):
        if diode_alpha is not None and not 0 <= diode_alpha <= 1.0:
            raise ValueError("Diode relaxation factor should be between 0 and 1 (inclusive).")

        if diode_frequency is not None and diode_frequency <= 0:
            raise ValueError("Fixed diode frequency must be larger than zero.")

        self.diode_frequency = diode_frequency
        self.diode_alpha = diode_alpha

        diode_params = DiodeModel().fitted_params
        fixed_params = np.asarray([diode_frequency, diode_alpha])

        self.fitted_params = [
            parameter for fixed, parameter in zip(fixed_params, diode_params) if fixed is None
        ]
        self._fixed_params = {
            parameter.name: CalibrationParameter(
                parameter.description + " (fixed)", fixed, parameter.unit
            )
            for fixed, parameter in zip(fixed_params, diode_params)
            if fixed is not None
        }
        self._fitted_idx = [idx for idx, parameter in enumerate(fixed_params) if parameter is None]
        self._parameters = fixed_params

    def params(self):
        """Parameters that were fixed during the fitting"""
        return self._fixed_params

    def __call__(self, f, *pars):
        self._parameters[self._fitted_idx] = pars
        return g_diode(f, *self._parameters)


class PassiveCalibrationModel:
    """Model to fit data acquired during passive calibration.

    Passive calibration involves fitting a power spectral density using a physical model of
    Brownian motion of a bead inside an optical trap.

    The power spectrum calibration algorithm implemented here is based on a number of publications
    by the Flyvbjerg group at DTU [1]_ [2]_ [3]_ [4]_ [5]_ [6]_. Please refer to the
    :doc:`theory section</theory/force_calibration/force_calibration>` and
    :doc:`tutorial</tutorial/force_calibration>` on force calibration for more information on the
    calibration methods implemented.

    Parameters
    ----------
    bead_diameter : float
        Bead diameter [um].
    viscosity : float, optional
        Liquid viscosity [Pa*s].
        When omitted, the temperature will be used to look up the viscosity of water at that
        particular temperature.
    temperature : float, optional
        Liquid temperature [Celsius].
    hydrodynamically_correct : bool, optional
        Enable hydrodynamic correction.
    distance_to_surface : float, optional
        Distance from bead center to the surface [um].
        When specifying `None`, the model will use an approximation which is only suitable for
        measurements performed deep in bulk.
    rho_sample : float, optional
        Density of the sample [kg/m^3]. Only used when using hydrodynamic corrections.
    rho_bead : float, optional
        Density of the bead [kg/m^3]. Only used when using hydrodynamic corrections.
    fast_sensor : bool
        Fast sensor? Fast sensors do not have the diode effect included in the model.
    axial : bool
        Is this an axial force model?

    Raises
    ------
    ValueError
        If physical parameters are provided that are outside their sensible range.
    ValueError
        If the distance from the bead center to the surface is set smaller than the bead radius.
    ValueError
        If the hydrodynamically correct model is enabled, but the distance to the surface is
        specified below 0.75 times the bead diameter (this model is not valid so close to the
        surface).
    NotImplementedError
        If the hydrodynamically correct model is selected in conjunction with axial force
        calibration.

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

    Examples
    --------
    ::

        f = lk.File("passive_calibration.h5")
        force_slice = f.force1x

        # Decalibrate existing data
        volts = force_slice / force_slice.calibration[0]["Response (pN/V)"]

        power_spectrum = lk.calculate_power_spectrum(
            volts.data,
            sample_rate=78125,
            excluded_ranges=[[4400, 4500]],  # Exclude a noise peak
            num_points_per_block=350,
        )

        model = lk.PassiveCalibrationModel(
            bead_diameter=4.89,
            temperature=25,
            hydrodynamically_correct=True,  # Should use hydrodynamic model for big beads
        )

        fit = lk.fit_power_spectrum(power_spectrum, model)
        fit.plot()
    """

    def __name__(self):
        return "PassiveCalibrationModel"

    def __init__(
        self,
        bead_diameter,
        viscosity=None,
        temperature=20,
        hydrodynamically_correct=False,
        distance_to_surface=None,
        rho_sample=None,
        rho_bead=1060.0,
        fast_sensor=False,
        axial=False,
    ):
        if bead_diameter < 1e-2:
            raise ValueError(
                f"Invalid bead diameter specified {bead_diameter}. Bead diameter should be bigger "
                f"than 10^-2 um"
            )

        if distance_to_surface is not None and distance_to_surface < bead_diameter / 2.0:
            raise ValueError("Distance from bead center to surface is smaller than the bead radius")

        if viscosity is not None and viscosity <= 0.0003:
            raise ValueError("Viscosity must be higher than 0.0003 Pa*s")

        if not 5.0 < temperature < 90.0:
            raise ValueError("Temperature must be between 5 and 90 Celsius")

        self.viscosity = viscosity if viscosity is not None else viscosity_of_water(temperature)
        self.temperature = temperature
        self.bead_diameter = bead_diameter
        self.drag_coeff = sphere_friction_coefficient(self.viscosity, self.bead_diameter * 1e-6)
        self._filter = NoFilter() if fast_sensor else DiodeModel()
        self._drag_fieldname = "gamma_0"
        self._drag_description = "Theoretical bulk drag coefficient"

        self.axial = axial
        self.hydrodynamically_correct = hydrodynamically_correct
        # Note that the default is set to None because in the future we may want to provide an
        # estimate of rho_sample based on temperature. If we pin this to a value already, this
        # would become a breaking change.
        self.rho_sample = rho_sample if rho_sample is not None else 997.0
        self.rho_bead = rho_bead
        self.distance_to_surface = distance_to_surface

        if hydrodynamically_correct:
            if self.axial:
                raise NotImplementedError(
                    "No hydrodynamically correct axial force model is currently available."
                )

            # The hydrodynamically correct model already accounts for the effect of a nearby wall.
            # Here the drag coefficient in the model represents the bulk drag coefficient.
            self._drag_correction_factor = 1
            # This model is only valid up to l/R < 1.5 [6] so throw in case that is violated.
            if (
                distance_to_surface is not None
                and distance_to_surface / (self.bead_diameter / 2) < 1.5
            ):
                raise ValueError(
                    "The hydrodynamically correct model is only valid for distances to the surface "
                    "larger than 0.75 times the bead diameter. For distances closer to the "
                    "surface, turn off the hydrodynamically correct model."
                )

            if rho_sample is not None and rho_sample < 100.0:
                raise ValueError("Density of the sample cannot be below 100 kg/m^3")

            if rho_bead < 100.0:
                raise ValueError("Density of the bead cannot be below 100 kg/m^3")

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
            if distance_to_surface:
                args = (distance_to_surface * 1e-6, bead_diameter * 1e-6 / 2.0)
                self._drag_correction_factor = (
                    brenner_axial(*args) if self.axial else faxen_factor(*args)
                )
            else:
                self._drag_correction_factor = 1

            self._passive_power_spectrum_model = passive_power_spectrum_model

    def _calculate_power_spectral_density(self, f, fc, diffusion_constant, *filter_params):
        physical_spectrum = self._passive_power_spectrum_model(f, fc, diffusion_constant)
        return physical_spectrum * self._filter(f, *filter_params)

    def __call__(self, f, fc, diffusion_constant, *filter_params) -> np.ndarray:
        return self._calculate_power_spectral_density(f, fc, diffusion_constant, *filter_params)

    def _alias_model(self, sample_rate, num_aliases):
        """Include effects of aliasing into the model

        Parameters
        ----------
        sample_rate : int
            Sample rate in Hz
        num_aliases : int
            Number of aliases to take into account.
        """
        new_model = copy(self)
        new_model._calculate_power_spectral_density = alias_spectrum(
            new_model._calculate_power_spectral_density, sample_rate, num_aliases
        )

        return new_model

    @property
    def _drag(self):
        """Returns the corrected drag coefficient

        Note that the hydrodynamic model already has a surface dependent drag coefficient baked in
        and therefore no correction is necessary (in this case self._drag_correction_factor is set
        to 1).
        """
        return self.drag_coeff * self._drag_correction_factor

    def _set_drag(
        self,
        drag,
        new_name="gamma_ex_lateral",
        new_description="Bulk drag coefficient from lateral calibration",
    ):
        """Set the bulk drag parameter used for this calibration."""
        self.drag_coeff = drag
        self._drag_fieldname = new_name
        self._drag_description = new_description

    def calibration_parameters(self) -> dict:
        hydrodynamic_parameters = (
            {
                "Sample density": CalibrationParameter(
                    "Density of the sample", self.rho_sample, "kg/m**3"
                ),
                "Bead density": CalibrationParameter(
                    "Density of bead material", self.rho_bead, "kg/m**3"
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
            "Distance to surface": CalibrationParameter(
                "Distance from bead center to surface", self.distance_to_surface, "um"
            ),
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
    ) -> dict:
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
        import scipy.constants

        # diameter [um] -> [m]
        temperature_k = scipy.constants.convert_temperature(self.temperature, "C", "K")

        # Distance response (Rd) needs to be output in um/V (m -> um or 1e6)
        distance_response = (
            np.sqrt(scipy.constants.k * temperature_k / self._drag / diffusion_constant_volts) * 1e6
        )

        # Stiffness is output in pN/nm (N/m -> pN/nm or 1e12 / 1e9 = 1e3)
        kappa = 2 * np.pi * self._drag * fc * 1e3

        # Force response (Rf) is output in pN/V. Rd [um/V], stiffness [pN/nm]: um -> nm = 1e3
        force_response = distance_response * kappa * 1e3

        return {
            "Rd": CalibrationParameter("Distance response", distance_response, "um/V"),
            "kappa": CalibrationParameter("Trap stiffness", kappa, "pN/nm"),
            "Rf": CalibrationParameter("Force response", force_response, "pN/V"),
            self._drag_fieldname: CalibrationParameter(
                self._drag_description, self.drag_coeff, "kg/s"
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

    In active calibration, we oscillate the stage with a known frequency and amplitude. This
    introduces an extra peak in the power spectrum which allows the trap to be calibrated with
    fewer assumptions. This trades some precision for accuracy.

    The power spectrum calibration algorithm implemented here is based on [1]_ [2]_ [3]_ [4]_ [5]_
    [6]_. Please refer to the :doc:`theory section</theory/force_calibration/force_calibration>` and
    :doc:`tutorial</tutorial/force_calibration>` on force calibration for more information on the
    calibration methods implemented.

    Parameters
    ----------
    driving_data : numpy.ndarray
        Array of driving data.
    force_voltage_data : numpy.ndarray
        Uncalibrated force data in volts.
    sample_rate : float
        Sample rate at which the signals were acquired.
    bead_diameter : float
        Bead diameter [um].
    driving_frequency_guess : float
        Guess of the driving frequency.
    viscosity : float, optional
        Liquid viscosity [Pa*s].
        When omitted, the temperature will be used to look up the viscosity of water at that
        particular temperature.
    temperature : float, optional
        Liquid temperature [Celsius].
    num_windows : int, optional
        Number of windows to average for the uncalibrated force. Using a larger number of
        windows potentially increases spectral bleed from adjacent peaks, but may be useful when
        the SNR is low.
    hydrodynamically_correct : bool, optional
        Enable hydrodynamically correct model.
    distance_to_surface : float, optional
        Distance from bead center to the surface [um]
        When specifying `None`, the model will use an approximation which is only suitable for
        measurements performed deep in bulk.
    rho_sample : float, optional
        Density of the sample [kg/m**3]. Only used when using hydrodynamically correct model.
    rho_bead : float, optional
        Density of the bead [kg/m**3]. Only used when using hydrodynamically correct model.
    fast_sensor : bool
        Fast sensor? Fast sensors do not have the diode effect included in the model.

    Attributes
    ----------
    driving_frequency : float
        Estimated driving frequency [Hz].
    driving_amplitude : float
        Estimated driving amplitude [m].
    viscosity : float, optional
        Liquid viscosity [Pa*s].

    Raises
    ------
    ValueError
        If physical parameters are provided that are outside their sensible range.
    ValueError
        If the distance from the bead center to the surface is set smaller than the bead radius.
    ValueError
        If the hydrodynamically correct model is enabled, but the distance to the surface is
        specified below 0.75 times the bead diameter (this model is not valid so close to the
        surface).
    NotImplementedError
        If the hydrodynamically correct model is selected in conjunction with axial force
        calibration.
    RuntimeError
        If the driving peak can't be found near the guess of its frequency.

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

    Examples
    --------
    ::

        f = lk.File("near_surface_active_calibration.h5")

        force_slice = f.force1x

        # Decalibrate existing data
        volts = force_slice / force_slice.calibration[0]["Response (pN/V)"]

        power_spectrum = lk.calculate_power_spectrum(
            volts.data,
            sample_rate=78125,
            excluded_ranges=[[4400, 4500]],  # Exclude a noise peak
            num_points_per_block=350,
        )

        model = lk.ActiveCalibrationModel(
            f["Nanostage position"]["X"].data,  # Driving data
            volts.data,  # Position sensitive detector data
            temperature=25,
            sample_rate=78125,
            bead_diameter=4.89,
            driving_frequency_guess=37,  # Have to provide a guess for the frequency
            hydrodynamically_correct=True,  # Big bead, so use hydrodynamic model
            distance_to_surface=10,  # Experiment performed 10 microns from surface
        )

        print(model.driving_frequency)  # Verify correct frequency determination

        fit = lk.fit_power_spectrum(power_spectrum, model)
        fit.plot()
    """

    def __name__(self) -> str:
        return "ActiveCalibrationModel"

    def __init__(
        self,
        driving_data,
        force_voltage_data,
        sample_rate,
        bead_diameter,
        driving_frequency_guess,
        viscosity=None,
        temperature=20,
        num_windows=5,
        hydrodynamically_correct=False,
        distance_to_surface=None,
        rho_sample=None,
        rho_bead=1060.0,
        fast_sensor=False,
    ):
        super().__init__(
            bead_diameter,
            viscosity,
            temperature,
            hydrodynamically_correct,
            distance_to_surface,
            rho_sample,
            rho_bead,
            fast_sensor,
            False,
        )
        self.driving_frequency_guess = driving_frequency_guess
        self.sample_rate = sample_rate
        self.num_windows = num_windows
        self._measured_drag_fieldname = "gamma_ex"

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

    def calibration_parameters(self) -> dict:
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
    ) -> dict:
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
        import scipy.constants

        reference_peak = self(self.driving_frequency, fc, diffusion_constant_volts, *filter_params)

        # Equation 14 from [6]
        power_exp = (self._response_power_density - reference_peak) * self._frequency_bin_width

        # Equation 12 from [6]
        distance_response = np.sqrt(self._theoretical_driving_power(fc) / power_exp)  # m/V

        # Equation 16 from [6]
        temperature_kelvin = scipy.constants.convert_temperature(self.temperature, "C", "K")
        k_temperature = scipy.constants.k * temperature_kelvin
        measured_drag_coeff = k_temperature / (
            distance_response**2 * diffusion_constant_volts
        )  # kg/s

        kappa = 2.0 * np.pi * fc * measured_drag_coeff  # N/m

        force_response = distance_response * kappa  # N/V

        return {
            "Rd": CalibrationParameter("Distance response", distance_response * 1e6, "um/V"),
            "kappa": CalibrationParameter("Trap stiffness", kappa * 1e3, "pN/nm"),
            "Rf": CalibrationParameter("Force response", force_response * 1e12, "pN/V"),
            self._drag_fieldname: CalibrationParameter(
                self._drag_description, self.drag_coeff, "kg/s"
            ),
            self._measured_drag_fieldname: CalibrationParameter(
                "Measured bulk drag coefficient",
                measured_drag_coeff / self._drag_correction_factor,
                "kg/s",
            ),
            "driving_amplitude": CalibrationParameter(
                "Driving amplitude",
                self.driving_amplitude * 1e6,
                "um",
            ),
            "driving_frequency": CalibrationParameter(
                "Driving frequency",
                self.driving_frequency,
                "Hz",
            ),
            "driving_power": CalibrationParameter(
                "Experimentally determined power in the spike observed on the positional power "
                "spectrum",
                power_exp,
                "V^2",
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
