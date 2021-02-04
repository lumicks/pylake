import pytest
import numpy as np
from lumicks.pylake.force_calibration.detail.power_models import *
from lumicks.pylake.force_calibration.power_spectrum_calibration import CalibrationSettings
from lumicks.pylake.force_calibration.detail.power_spectrum import PowerSpectrum


def test_friction_coefficient():
    assert np.allclose(sphere_friction_coefficient(5, 1), 3 * np.pi * 5 * 1)


def test_spectrum(reference_models):
    assert np.allclose(FullPSFitModel.P(np.arange(10000), 1000, 1e9, 10000, 0.5),
                       reference_models.lorentzian_filtered(np.arange(10000), 1000, 1e9, 0.5, 10000))


def test_spectrum_parameter_scaling(reference_models):
    f = np.arange(10000)
    scaling = [2.0, 3.0, 4.0, 5.0]
    scaled_psc = FullPSFitModel(scaling)

    alpha = 0.5
    inverted_alpha = np.sqrt(((1.0 / alpha) ** 2.0 - 1.0) / scaling[3]**2)

    assert np.allclose(scaled_psc(f, 1000 / scaling[0], 1e9 / scaling[1], 10000 / scaling[2], inverted_alpha),
                       reference_models.lorentzian_filtered(np.arange(10000), 1000, 1e9, 0.5, 10000))


@pytest.mark.parametrize(
    "corner_frequency,diffusion_constant,num_samples,sigma_fc,sigma_diffusion",
    [
        [1000, 1e-9, 30000, 21.113382377506746, 1.1763968470146817e-11],
        [1500, 1.2e-9, 50000, 28.054469036154266, 1.5342555193009045e-11],
        [1500, 1.2e-9, 10000, 28.068761746382837, 1.5365711725327977e-11],
        [1500, 1.2e-9, 10000, 28.068761746382837, 1.5365711725327977e-11],
        [1500, 1.2e-9, 10000, 28.068761746382837, 1.5365711725327977e-11],
        [1500, 1.2e-9, 10000, 28.068761746382837, 1.5365711725327977e-11],
        [1000, 1e-9, 30000, 21.113382377506746, 1.1763968470146817e-11],
        [1000, 1e-9, 30000, 21.113382377506746, 1.1763968470146817e-11],
    ],
)
def test_fit_analytic(reference_models, corner_frequency, diffusion_constant, num_samples, sigma_fc, sigma_diffusion):
    power_spectrum = reference_models.lorentzian(np.arange(0, num_samples), corner_frequency, diffusion_constant)
    data = np.fft.irfft(np.sqrt(np.abs(power_spectrum))) * (num_samples - 1) * 2

    num_points_per_block = 20
    fit_range = (0, 15000)
    ps = PowerSpectrum(data, sampling_rate=len(data))
    ps = ps.in_range(*fit_range)
    ps = ps.block_averaged(num_blocks=ps.P.size // num_points_per_block)

    fit = fit_analytical_lorentzian(ps.in_range(1e1, 1e4))

    assert np.allclose(fit.fc, corner_frequency)
    assert np.allclose(fit.D, diffusion_constant)
    assert np.allclose(fit.sigma_fc, sigma_fc)
    assert np.allclose(fit.sigma_D, sigma_diffusion)
