from lumicks.pylake.force_calibration import power_spectrum_calibration as psc
import numpy as np
import scipy as sp
import os
import pytest


def lorentzian(f, fc, diffusion_constant):
    return diffusion_constant / (2.0 * np.pi ** 2) / (fc ** 2 + f ** 2)


def lorentzian_filtered(f, fc, diffusion_constant, alpha, f_diode):
    return lorentzian(f, fc, diffusion_constant) * (alpha ** 2 + (1.0 - alpha ** 2) / (1.0 + (f / f_diode) ** 2))


def lorentzian_td(corner_frequency, diffusion_constant, alpha, f_diode, num_samples):
    f = np.arange(0, num_samples)
    power_spectrum = lorentzian_filtered(f, corner_frequency, diffusion_constant, alpha, f_diode)
    data = np.fft.irfft(np.sqrt(np.abs(power_spectrum))) * (num_samples - 1) * 2
    return data, len(data)


def test_friction_coefficient():
    assert np.allclose(psc.sphere_friction_coefficient(5, 1), 3 * np.pi * 5 * 1)


def test_calibration_parameters():
    params = psc.CalibrationParameters(10, temperature=30)
    assert params.bead_diameter == 10
    assert params.viscosity == 1.002e-3
    assert params.temperature == 30

    with pytest.raises(TypeError):
        psc.CalibrationParameters(10, invalid_parameter=5)

    assert np.allclose(params.temperature_K(), 303.15)


def test_calibration_settings():
    settings = psc.CalibrationSettings(n_points_per_block=15)
    assert settings.n_points_per_block == 15
    assert settings.fit_range == (1e2, 23e3)
    assert settings.analytical_fit_range == (1e1, 1e4)
    assert settings.ftol == 1e-7
    assert settings.maxfev == 10000

    with pytest.raises(TypeError):
        psc.CalibrationSettings(invalid_parameter=5)


def test_input_validation_power_spectrum_calibration():
    params = psc.CalibrationParameters(1)

    # Wrong dimensions
    with pytest.raises(TypeError):
        psc.PowerSpectrumCalibration(data=np.array([[1, 2, 3], [1, 2, 3]]), sampling_rate=78125, params=params)

    # Wrong type
    with pytest.raises(TypeError):
        psc.PowerSpectrumCalibration(data="bloop", sampling_rate=78125, params=params)

    with pytest.raises(TypeError):
        psc.PowerSpectrumCalibration(data=np.array([1, 2, 3]), sampling_rate=78125, params="invalid")

    with pytest.raises(TypeError):
        psc.PowerSpectrumCalibration(data=np.array([1, 2, 3]), sampling_rate=78125, params=params, settings="invalid")


def test_default_calibration_settings():
    power_spectrum = psc.PowerSpectrumCalibration(data=np.array([1, 2, 3]), sampling_rate=78125,
                                                  params=psc.CalibrationParameters(1))

    attrs = ['n_points_per_block', 'fit_range', 'analytical_fit_range', 'ftol', 'maxfev']
    defaults = psc.CalibrationSettings()
    for attr in attrs:
        assert getattr(power_spectrum.settings, attr) == getattr(defaults, attr)


def test_spectrum():
    assert np.allclose(psc.FullPSFitModel.P(np.arange(10000), 1000, 1e9, 10000, 0.5),
                       lorentzian_filtered(np.arange(10000), 1000, 1e9, 0.5, 10000))


def test_spectrum_parameter_scaling():
    f = np.arange(10000)
    scaling = [2.0, 3.0, 4.0, 5.0]
    scaled_psc = psc.FullPSFitModel(scaling)

    alpha = 0.5
    inverted_alpha = np.sqrt(((1.0 / alpha) ** 2.0 - 1.0) / scaling[3]**2)

    assert np.allclose(scaled_psc(f, 1000 / scaling[0], 1e9 / scaling[1], 10000 / scaling[2], inverted_alpha),
                       lorentzian_filtered(np.arange(10000), 1000, 1e9, 0.5, 10000))


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
def test_fit_analytic(corner_frequency, diffusion_constant, num_samples, sigma_fc, sigma_diffusion):
    power_spectrum = lorentzian(np.arange(0, num_samples), corner_frequency, diffusion_constant)
    data = np.fft.irfft(np.sqrt(np.abs(power_spectrum))) * (num_samples - 1) * 2

    ps = psc.PowerSpectrum(data, sampling_rate=len(data))
    settings = psc.CalibrationSettings(fit_range=(0, 15000), n_points_per_block=20)

    ps = ps.in_range(0, 15000)
    ps = ps.block_averaged(n_blocks=ps.P.size // settings.n_points_per_block)

    fit = psc.fit_analytical_lorentzian(ps.in_range(*settings.analytical_fit_range))

    assert np.allclose(fit.fc, corner_frequency)
    assert np.allclose(fit.D, diffusion_constant)
    assert np.allclose(fit.sigma_fc, sigma_fc)
    assert np.allclose(fit.sigma_D, sigma_diffusion)


def test_calibration_result():
    with pytest.raises(TypeError):
        psc.CalibrationResults(invalid=5)

    calibration_result = psc.CalibrationResults(fc=5)
    assert calibration_result.is_success() is True
    setattr(calibration_result, "error", "error!")
    assert calibration_result.is_success() is False


@pytest.mark.parametrize(
    "corner_frequency,diffusion_constant,alpha,f_diode,num_samples,viscosity,bead_diameter,temperature,err_fc,err_d,"
    "err_f_diode,err_alpha,",
    [
        [1000, 1e-9, 0.5, 10000, 30000, 1.002e-3, 4.0, 20.0, 29.77266, 0.00000, 1239.06183, 0.05650019],
        [1500, 1.2e-9, 0.5, 10000, 50000, 1.002e-3, 4.0, 20.0, 47.21810, 0.00000, 1399.04982, 0.05896166],
        [1500, 1.2e-9, 0.5, 5000, 10000, 1.002e-3, 4.0, 20.0, 84.85429, 0.00000, 827.43118, 0.03077534],
        [1500, 1.2e-9, 0.5, 5000, 10000, 1.2e-3, 4.0, 20.0, 84.85429, 0.00000, 827.43118, 0.03077534],
        [1500, 1.2e-9, 0.5, 5000, 10000, 1.002e-3, 8.0, 20.0, 84.85429, 0.00000, 827.43118, 0.03077534],
        [1500, 1.2e-9, 0.5, 5000, 10000, 1.002e-3, 4.0, 34.0, 84.85429, 0.00000, 827.43118, 0.03077534],
        [1000, 1e-9, 0.5, 10000, 30000, 1.002e-3, 4.0, 20.0, 29.77266, 0.00000, 1239.06183, 0.05650019],
        [1000, 1e-9, 0.5, 10000, 30000, 1, 4.0, 20.0, 29.77266, 0.00000, 1239.06183, 0.05650019],
    ],
)
def test_good_fit_integration_test(
    corner_frequency,
    diffusion_constant,
    alpha,
    f_diode,
    num_samples,
    viscosity,
    bead_diameter,
    temperature,
    err_fc,
    err_d,
    err_f_diode,
    err_alpha,
):
    data, f_sample = lorentzian_td(corner_frequency, diffusion_constant, alpha, f_diode, num_samples)
    params = psc.CalibrationParameters(bead_diameter, temperature=temperature, viscosity=viscosity)
    settings = psc.CalibrationSettings(fit_range=(0, 15000), n_points_per_block=20)
    ps_calibration = psc.PowerSpectrumCalibration(data=data, sampling_rate=f_sample, params=params, settings=settings)

    ps_calibration.run_fit()

    assert np.allclose(ps_calibration.results.fc, corner_frequency, rtol=1e-4)
    assert np.allclose(ps_calibration.results.D, diffusion_constant, rtol=1e-5)
    assert np.allclose(ps_calibration.results.alpha, alpha, rtol=1e-4)
    assert np.allclose(ps_calibration.results.f_diode, f_diode, rtol=1e-4)

    gamma = psc.sphere_friction_coefficient(viscosity, bead_diameter * 1e-6)
    kappa_true = 2.0 * np.pi * gamma * corner_frequency * 1e3
    rd_true = (
        np.sqrt(sp.constants.k * sp.constants.convert_temperature(temperature, "C", "K") / gamma / diffusion_constant)
        * 1e6
    )
    assert np.allclose(ps_calibration.results.kappa, kappa_true, rtol=1e-4)
    assert np.allclose(ps_calibration.results.Rd, rd_true, rtol=1e-4)
    assert np.allclose(ps_calibration.results.Rf, rd_true * kappa_true * 1e3, rtol=1e-4)
    assert np.allclose(ps_calibration.results.chi_squared_per_deg, 0)  # Noise free

    assert np.allclose(ps_calibration.results.err_fc, err_fc)
    assert np.allclose(ps_calibration.results.err_D, err_d)
    assert np.allclose(ps_calibration.results.err_f_diode, err_f_diode)
    assert np.allclose(ps_calibration.results.err_alpha, err_alpha)

    assert ps_calibration.results.is_success() is True

    # This field actually never gets set, which is very strange. But for exact parity, we keep it.
    setattr(ps_calibration.results, "error", "Error!")
    assert ps_calibration.results.is_success() is False


def test_bad_calibration_result_arg():
    with pytest.raises(TypeError):
        psc.CalibrationResults(bad_arg=5)


def test_bad_data():
    num_samples = 30000
    data = np.sin(.1*np.arange(num_samples))
    params = psc.CalibrationParameters(1, temperature=20, viscosity=0.0001)
    settings = psc.CalibrationSettings(fit_range=(0, 15000), n_points_per_block=20)

    ps_calibration = psc.PowerSpectrumCalibration(data=data, sampling_rate=num_samples, params=params, settings=settings)
    with pytest.raises(psc.CalibrationError):
        ps_calibration.run_fit()

    # What?? This doesn't seem good, but for parity, we test the current strange behaviour explicitly.
    assert ps_calibration.results.is_success() is True

def test_actual_spectrum():

    data = np.load(os.path.join(os.path.dirname(__file__), "reference_spectrum.npz"))
    reference_spectrum = data["arr_0"]
    params = psc.CalibrationParameters(4.4, temperature=20, viscosity=0.001002)
    settings = psc.CalibrationSettings(fit_range=(100.0, 23000.0), n_points_per_block=int(100))

    ps_calibration = psc.PowerSpectrumCalibration(data=reference_spectrum, sampling_rate=78125, params=params,
                                                  settings=settings)
    ps_calibration.run_fit()

    assert np.allclose(ps_calibration.results.D, 0.0018512665210876748, rtol=1e-4)
    assert np.allclose(ps_calibration.results.Rd, 7.253645956145265, rtol=1e-4)
    assert np.allclose(ps_calibration.results.Rf, 1243.9711315478219, rtol=1e-4)
    assert np.allclose(ps_calibration.results.kappa, 0.17149598134079505, rtol=1e-4)
    assert np.allclose(ps_calibration.results.alpha, 0.5006103727942776, rtol=1e-4)
    assert np.allclose(ps_calibration.results.backing, 66.4331056392512, rtol=1e-4)
    assert np.allclose(ps_calibration.results.chi_squared_per_deg, 1.063783302378645, rtol=1e-4)
    assert np.allclose(ps_calibration.results.err_fc, 32.23007993226726, rtol=1e-4)
    assert np.allclose(ps_calibration.results.err_D, 6.43082000774291e-05, rtol=1e-4)
    assert np.allclose(ps_calibration.results.err_alpha, 0.013141463933316694, rtol=1e-4)
    assert np.allclose(ps_calibration.results.err_f_diode, 561.6377089699399, rtol=1e-4)
