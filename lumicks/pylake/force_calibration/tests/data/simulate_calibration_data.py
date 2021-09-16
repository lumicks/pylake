import numpy as np
from functools import partial
import scipy.constants
from lumicks.pylake.force_calibration.detail.power_models import (
    sphere_friction_coefficient,
    passive_power_spectrum_model,
    g_diode,
)
from lumicks.pylake.force_calibration.detail.hydrodynamics import (
    passive_power_spectrum_model_hydro,
    theoretical_driving_power_hydrodynamics,
)


def power_model_to_time_series(sample_rate, num_points, power_spectral_density):
    """Generates channel data with a desired power spectral density.

    Function generates random uncorrelated noise and then multiplies it spectrally such that it
    would have a specific target spectral density. Returns time trace of a specified duration.

    Parameters
    ----------
    sample_rate : int
        Sample rate
    num_points : float
        Number of points in the desired trace.
    power_spectral_density : callable
        Function that takes frequencies and produces power spectral density in V^2/Hz at those
        points.
    """
    freq = np.fft.rfftfreq(num_points, 1 / sample_rate)
    noise = np.sqrt(sample_rate / 2) * np.random.randn(num_points)  # Flat spectrum at 1.0 V^2/Hz
    return np.fft.irfft(np.fft.rfft(noise) * np.sqrt(power_spectral_density(freq)))


def response_peak_ideal(corner_frequency, driving_frequency, driving_amplitude):
    """Spectral peak corresponding to the driving input (no hydrodynamic corrections).

    Eq. 13 in [1].

    [1] Tolić-Nørrelykke, S. F., Schäffer, E., Howard, J., Pavone, F. S., Jülicher, F., &
    Flyvbjerg, H. (2006). Calibration of optical tweezers with positional detection in the back
    focal plane. Review of scientific instruments, 77(10), 103101.
    """
    return driving_amplitude ** 2 / (2 * (1 + (corner_frequency / driving_frequency) ** 2))


def generate_active_calibration_test_data(
    duration,
    sample_rate,
    bead_diameter,
    stiffness,
    viscosity,
    temperature,
    pos_response_um_volt,
    driving_sinusoid,
    diode,
    hydrodynamically_correct=False,
    rho_sample=None,
    rho_bead=1060.0,
    distance_to_surface=None,
):
    """Generate test data to test active calibration.

    Note: This generation does not get the phase information right (but this is not used for
    spectral calibration).

    Parameters
    ----------
    duration : float
        Time [s]
    sample_rate : int
        Sampling rate [Hz]
    bead_diameter : float
        Bead diameter [micron]
    stiffness : float
        Spring constant of the trap [pN/nm]
    viscosity : float
        Viscosity [Pa*s]
    temperature : float
        Temperature [C]
    pos_response_um_volt : float
        Response [um/V], also denoted in papers as Rd
    driving_sinusoid : tuple of floats
        Parameters for the driving input.
        Amplitude [nm] and frequency of active [Hz] calibration stage movement
    diode : tuple of floats
        Diode parameters:
        Alpha of the diode response that is instantaneous, ranges from 0 to 1 [-]
        Corner frequency of the filtering effect of the PSD [Hz]
    hydrodynamically_correct : bool, optional
        Enable hydrodynamically correct model.
    rho_sample : float, optional
        Density of the sample. Only used when using hydrodynamically correct model.
    rho_bead : float, optional
        Density of the bead. Only used when using hydrodynamically correct model.
    distance_to_surface : float, optional
        Distance from bead center to the surface. Only used when using hydrodynamically correct
        model. None uses the approximation valid for deep in bulk.
    """
    pos_response_m_volt = pos_response_um_volt / 1e6
    gamma_0 = sphere_friction_coefficient(viscosity, bead_diameter * 1e-6)  # Ns/m
    diffusion_physical = scipy.constants.k * (temperature + 273.15) / gamma_0  # m^2/s
    diffusion_volt = diffusion_physical / pos_response_m_volt / pos_response_m_volt  # V^2/s
    stiffness_si = stiffness * 1e-3  # N/m
    fc = stiffness_si / (2.0 * np.pi * gamma_0)  # Hz
    driving_amplitude, driving_frequency = driving_sinusoid

    basic_pars = {
        "fc": fc,
        "diffusion_constant": diffusion_volt,
    }
    hydro_pars = {
        "gamma0": gamma_0,
        "bead_radius": bead_diameter * 1e-6 / 2,
        "rho_sample": rho_sample,
        "rho_bead": rho_bead,
        "distance_to_surface": None if distance_to_surface is None else distance_to_surface * 1e-6,
    }

    power_spectrum_model = (
        partial(passive_power_spectrum_model_hydro, **basic_pars, **hydro_pars)
        if hydrodynamically_correct
        else partial(passive_power_spectrum_model, **basic_pars)
    )

    def filtered_power_spectrum(f):
        return power_spectrum_model(f) * g_diode(f, diode[1], diode[0])

    response_peak = (
        partial(theoretical_driving_power_hydrodynamics, **hydro_pars)
        if hydrodynamically_correct
        else response_peak_ideal
    )

    p_response_m_squared = response_peak(fc, driving_frequency, driving_amplitude * 1e-9)
    p_response_volts = np.sqrt(p_response_m_squared) / pos_response_m_volt

    num_points = 2 * (sample_rate * duration // 2)  # Has to be multiple of two for FFT
    time = np.arange(num_points) / sample_rate
    driving_sine = np.sin(2.0 * np.pi * driving_frequency * time)
    nano_stage = driving_amplitude * 1e-9 * driving_sine

    # Center to peak amplitude is given by sqrt(2) * RMS Voltage
    psd_component = np.sqrt(2) * p_response_volts * driving_sine

    return (
        power_model_to_time_series(sample_rate, num_points, filtered_power_spectrum)
        + psd_component,
        nano_stage * 1e6,
    )
