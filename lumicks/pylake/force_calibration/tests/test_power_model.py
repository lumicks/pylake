import pytest
from lumicks.pylake.force_calibration.detail.power_models import *
from lumicks.pylake.force_calibration.power_spectrum import PowerSpectrum
from lumicks.pylake.force_calibration.power_spectrum_calibration import guess_f_diode_initial_value


def test_friction_coefficient():
    np.testing.assert_allclose(sphere_friction_coefficient(5, 1), 3 * np.pi * 5 * 1)


def test_spectrum(reference_models):
    np.testing.assert_allclose(
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

    np.testing.assert_allclose(
        scaled_psc(f, fc / initials[0], diff / initials[1], diode / initials[2], a_scaled),
        reference_models.lorentzian_filtered(f, fc, diff, alpha, diode),
    )

    # Test whether unity values in scaled-world indeed lead to the correct initial parameters
    np.testing.assert_allclose(scaled_psc.scale_params((1.0, 1.0, 1.0, 1.0)), initials)
    np.testing.assert_allclose(
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
    data = np.fft.irfft(np.sqrt(np.abs(power_spectrum) * 0.5), norm="forward")

    num_points_per_block = 20
    fit_range = (0, 15000)
    ps = PowerSpectrum(data, sample_rate=len(data))
    ps = ps.in_range(*fit_range)
    ps = ps.downsampled_by(num_points_per_block)

    fit = fit_analytical_lorentzian(ps.in_range(1e1, 1e4))

    np.testing.assert_allclose(fit.fc, corner_frequency, rtol=1e-5)
    np.testing.assert_allclose(fit.D, diffusion_constant, rtol=1e-5, atol=0)
    np.testing.assert_allclose(fit.sigma_fc, sigma_fc)
    np.testing.assert_allclose(fit.sigma_D, sigma_diffusion)


@pytest.mark.parametrize(
    "corner_frequency,diffusion_constant,num_samples,alpha,f_diode,f_diode_est",
    [
        [4000, 1e-9, 78125, 0.0, 16000.0, 16000.0],
        [5000, 1.2e-9, 78125, 0.0, 15000.0, 15000.0],
    ],
)
def test_guess_f_diode_guess(
    reference_models, corner_frequency, diffusion_constant, num_samples, alpha, f_diode, f_diode_est
):
    data, f_sample = reference_models.lorentzian_td(
        corner_frequency, diffusion_constant, alpha, f_diode, num_samples // 2
    )
    ps = PowerSpectrum(data, sample_rate=f_sample)
    np.testing.assert_allclose(
        guess_f_diode_initial_value(ps, corner_frequency, diffusion_constant), f_diode_est
    )


def test_fit_analytic_curve():
    ps = PowerSpectrum([3, 3, 4, 5, 1, 3, 2, 4, 5, 2], 100)
    ref = [0.079276, 0.077842, 0.073833, 0.067997, 0.061221, 0.054269]
    fit = fit_analytical_lorentzian(ps)
    np.testing.assert_allclose(fit.ps_fit.frequency, np.arange(0, 60, 10))
    np.testing.assert_allclose(fit.ps_fit.power, ref, rtol=1e-5)
