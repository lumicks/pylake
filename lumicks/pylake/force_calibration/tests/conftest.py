import numpy as np
import pytest

from .data.simulate_calibration_data import generate_active_calibration_test_data


class ReferenceModels:
    @staticmethod
    def lorentzian(f, fc, diffusion_constant):
        # Lorentzian in V^2/Hz
        return diffusion_constant / (np.pi**2) / (fc**2 + f**2)

    @staticmethod
    def lorentzian_filtered(f, fc, diffusion_constant, alpha, f_diode):
        return ReferenceModels.lorentzian(f, fc, diffusion_constant) * (
            alpha**2 + (1.0 - alpha**2) / (1.0 + (f / f_diode) ** 2)
        )

    @staticmethod
    def lorentzian_td(corner_frequency, diffusion_constant, alpha, f_diode, num_samples):
        f = np.arange(0, num_samples)
        power_spectrum = ReferenceModels.lorentzian_filtered(
            f, corner_frequency, diffusion_constant, alpha, f_diode
        )
        data = np.fft.irfft(np.sqrt(np.abs(power_spectrum * 0.5)), norm="forward")
        return data, len(data)


@pytest.fixture
def reference_models():
    return ReferenceModels()


@pytest.fixture
def integration_test_parameters():
    return {
        "bead_diameter": 1.03,
        "viscosity": 1.1e-3,
        "temperature": 25,
        "hydrodynamically_correct": True,
        "rho_sample": 997.0,
        "rho_bead": 1040.0,
        "distance_to_surface": 1.51 * 1.03 / 2,
    }, {
        "sample_rate": 78125,
        "stiffness": 0.1,
        "pos_response_um_volt": 0.618,
        "driving_sinusoid": (500, 31.95633),
        "diode": (0.4, 15000),
    }


@pytest.fixture
def mack_parameters():
    return {
        "wavelength_nm": 1064.0,
        "refractive_index_medium": 1.333,
        "surface_position": 101.65754300557573,
        "displacement_sensitivity": 9.724913160609043,
        "intensity_amplitude": -0.10858224326787835,
        "intensity_phase_shift": 1.6535670092299886,
        "intensity_decay_length": 0.308551871490813,
        "scattering_polynomial_coeffs": [
            -0.043577454353825644,
            0.22333743993836863,
            -0.33331150250090585,
            0.1035148152731559,
        ],
        "focal_shift": 0.921283446497108,
        "nonlinear_shift": 0.0,
    }


@pytest.fixture(scope="session")
def active_calibration_surface_data():
    shared_pars = {
        "bead_diameter": 1.03,
        "viscosity": 1.1e-3,
        "temperature": 25,
        "rho_sample": 997.0,
        "rho_bead": 1040.0,
        "distance_to_surface": 1.03 / 2 + 500e-3,
    }

    sim_pars = {
        "sample_rate": 78125,
        "stiffness": 0.1,
        "pos_response_um_volt": 0.618,
        "driving_sinusoid": (500, 31.95633),
        "diode": (0.4, 15000),
    }

    np.random.seed(10071985)
    volts, stage = generate_active_calibration_test_data(
        10, hydrodynamically_correct=True, **sim_pars, **shared_pars
    )

    active_pars = {
        "force_voltage_data": volts,
        "driving_data": stage,
        "sample_rate": 78125,
        "driving_frequency_guess": 32,
    }

    return shared_pars, sim_pars, active_pars
