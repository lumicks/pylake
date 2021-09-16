import pytest
from lumicks.pylake.force_calibration.detail.power_models import *
from lumicks.pylake.force_calibration.power_spectrum import PowerSpectrum


def test_friction_coefficient():
    np.testing.assert_allclose(sphere_friction_coefficient(5, 1), 3 * np.pi * 5 * 1)


def test_spectrum(reference_models):
    f = np.arange(10000)
    np.testing.assert_allclose(
        passive_power_spectrum_model(f, 1000, 1e9) * g_diode(f, 10000, 0.5),
        reference_models.lorentzian_filtered(np.arange(10000), 1000, 1e9, 0.5, 10000),
    )


def test_spectrum_parameter_scaling(reference_models):
    f = np.arange(10000)
    initials = np.array([2.0, 3.0, 4.0, 5.0])
    def filtered_power_spectrum(f, fc, diff, f_diode, alpha): 
        return passive_power_spectrum_model(f, fc, diff) * g_diode(f, f_diode, alpha)
    scaled_psc = ScaledModel(filtered_power_spectrum, initials)

    fc, diff, alpha, diode = 1000, 1e9, 0.5, 10000

    np.testing.assert_allclose(
        scaled_psc(
            f, fc / initials[0], diff / initials[1], diode / initials[2], alpha / initials[3]
        ),
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


def test_fit_analytic_curve():
    ps = PowerSpectrum([3, 3, 4, 5, 1, 3, 2, 4, 5, 2], 100)
    ref = [0.079276, 0.077842, 0.073833, 0.067997, 0.061221, 0.054269]
    fit = fit_analytical_lorentzian(ps)
    np.testing.assert_allclose(fit.ps_fit.frequency, np.arange(0, 60, 10))
    np.testing.assert_allclose(fit.ps_fit.power, ref, rtol=1e-5)
