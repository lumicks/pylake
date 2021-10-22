import pytest
import numpy as np


class ReferenceModels:
    @staticmethod
    def lorentzian(f, fc, diffusion_constant):
        # Lorentzian in V^2/Hz
        return diffusion_constant / (np.pi ** 2) / (fc ** 2 + f ** 2)

    @staticmethod
    def lorentzian_filtered(f, fc, diffusion_constant, alpha, f_diode):
        return ReferenceModels.lorentzian(f, fc, diffusion_constant) * (
            alpha ** 2 + (1.0 - alpha ** 2) / (1.0 + (f / f_diode) ** 2)
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
