import pytest
import numpy as np


class ReferenceModels:
    @staticmethod
    def lorentzian(f, fc, diffusion_constant):
        return diffusion_constant / (2.0 * np.pi ** 2) / (fc ** 2 + f ** 2)

    @staticmethod
    def lorentzian_filtered(f, fc, diffusion_constant, alpha, f_diode):
        return ReferenceModels.lorentzian(f, fc, diffusion_constant) * \
               (alpha ** 2 + (1.0 - alpha ** 2) / (1.0 + (f / f_diode) ** 2))

    @staticmethod
    def lorentzian_td(corner_frequency, diffusion_constant, alpha, f_diode, num_samples):
        f = np.arange(0, num_samples)
        power_spectrum = ReferenceModels.lorentzian_filtered(f, corner_frequency, diffusion_constant, alpha, f_diode)
        data = np.fft.irfft(np.sqrt(np.abs(power_spectrum))) * (num_samples - 1) * 2
        return data, len(data)


@pytest.fixture
def reference_models():
    return ReferenceModels()
