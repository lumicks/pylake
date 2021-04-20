import pytest
from lumicks.pylake.force_calibration.detail.power_models import *
from lumicks.pylake.force_calibration.power_spectrum import PowerSpectrum


def test_friction_coefficient():
    assert np.allclose(sphere_friction_coefficient(5, 1), 3 * np.pi * 5 * 1)


def test_spectrum(reference_models):
    assert np.allclose(
        passive_power_spectrum_model(np.arange(10000), 1000, 1e9, 10000, 0.5),
        reference_models.lorentzian_filtered(np.arange(10000), 1000, 1e9, 0.5, 10000),
    )


def test_spectrum_parameter_scaling(reference_models):
    from ..detail.power_models import _convert_to_alpha, _convert_to_a

    f = np.arange(10000)
    a_true = 5.0
    initials = [2.0, 3.0, 4.0, _convert_to_alpha(a_true)]
    scaled_psc = ScaledModel(passive_power_spectrum_model, initials)

    fc, diff, alpha, diode = 1000, 1e9, 0.5, 10000
    a_scaled = _convert_to_a(alpha) / a_true

    assert np.allclose(
        scaled_psc(f, fc / initials[0], diff / initials[1], diode / initials[2], a_scaled),
        reference_models.lorentzian_filtered(f, fc, diff, alpha, diode),
    )

    # Test whether unity values in scaled-world indeed lead to the correct initial parameters
    assert np.allclose(scaled_psc.scale_params((1.0, 1.0, 1.0, 1.0)), initials)
    assert np.allclose(
        scaled_psc(f, 1.0, 1.0, 1.0, 1.0),
        reference_models.lorentzian_filtered(f, initials[0], initials[1], initials[3], initials[2]),
        rtol=1e-6,
    )


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
def test_fit_analytic(
    reference_models, corner_frequency, diffusion_constant, num_samples, sigma_fc, sigma_diffusion
):
    power_spectrum = reference_models.lorentzian(
        np.arange(0, num_samples), corner_frequency, diffusion_constant
    )
    data = np.fft.irfft(np.sqrt(np.abs(power_spectrum))) * (num_samples - 1) * 2

    num_points_per_block = 20
    fit_range = (0, 15000)
    ps = PowerSpectrum(data, sample_rate=len(data))
    ps = ps.in_range(*fit_range)
    ps = ps.downsampled_by(num_points_per_block)

    fit = fit_analytical_lorentzian(ps.in_range(1e1, 1e4))

    assert np.allclose(fit.fc, corner_frequency)
    assert np.allclose(fit.D, diffusion_constant)
    assert np.allclose(fit.sigma_fc, sigma_fc)
    assert np.allclose(fit.sigma_D, sigma_diffusion)


def test_fit_analytic_curve():
    ps = PowerSpectrum([3, 3, 4, 5, 1, 3, 2, 4, 5, 2], 100)
    ref = [0.0396382, 0.0389208, 0.03691641, 0.03399826, 0.03061068, 0.02713453]
    fit = fit_analytical_lorentzian(ps)
    assert np.allclose(fit.ps_fit.frequency, np.arange(0, 60, 10))
    assert np.allclose(fit.ps_fit.power, ref)
