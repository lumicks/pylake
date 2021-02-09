from lumicks.pylake.force_calibration import power_spectrum_calibration as psc
from lumicks.pylake.force_calibration.calibration_models import PassiveCalibrationModel, sphere_friction_coefficient
import numpy as np
import scipy as sp
import os
import pytest


def test_model_parameters():
    params = PassiveCalibrationModel(10, temperature=30)
    assert params.bead_diameter == 10
    assert params.viscosity == 1.002e-3
    assert params.temperature == 30

    with pytest.raises(TypeError):
        PassiveCalibrationModel(10, invalid_parameter=5)


def test_calibration_settings():
    settings = psc.CalibrationSettings(ftol=1e-6)
    assert settings.analytical_fit_range == (1e1, 1e4)
    assert settings.ftol == 1e-6
    assert settings.maxfev == 10000

    with pytest.raises(TypeError):
        psc.CalibrationSettings(invalid_parameter=5)


def test_input_validation_power_spectrum_calibration():
    model = PassiveCalibrationModel(1)

    # Wrong dimensions
    with pytest.raises(TypeError):
        psc.fit_power_spectrum(data=np.array([[1, 2, 3], [1, 2, 3]]), sampling_rate=78125, model=model)

    # Wrong type
    with pytest.raises(TypeError):
        psc.fit_power_spectrum(data="bloop", sampling_rate=78125, model=model)

    with pytest.raises(TypeError):
        psc.fit_power_spectrum(data=np.array([1, 2, 3]), sampling_rate=78125, model="invalid")

    with pytest.raises(TypeError):
        psc.fit_power_spectrum(data=np.array([1, 2, 3]), sampling_rate=78125, model=model, settings="invalid")


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
    reference_models,
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
    data, f_sample = reference_models.lorentzian_td(corner_frequency, diffusion_constant, alpha, f_diode, num_samples)
    model = PassiveCalibrationModel(bead_diameter, temperature=temperature, viscosity=viscosity)
    power_spectrum = psc.calculate_power_spectrum(data, f_sample, fit_range=(0, 15000), num_points_per_block=20)
    ps_calibration = psc.fit_power_spectrum(power_spectrum=power_spectrum, model=model)

    assert np.allclose(ps_calibration.fc, corner_frequency, rtol=1e-4)
    assert np.allclose(ps_calibration.D, diffusion_constant, rtol=1e-5)
    assert np.allclose(ps_calibration.alpha, alpha, rtol=1e-4)
    assert np.allclose(ps_calibration.f_diode, f_diode, rtol=1e-4)

    gamma = sphere_friction_coefficient(viscosity, bead_diameter * 1e-6)
    kappa_true = 2.0 * np.pi * gamma * corner_frequency * 1e3
    rd_true = (
        np.sqrt(sp.constants.k * sp.constants.convert_temperature(temperature, "C", "K") / gamma / diffusion_constant)
        * 1e6
    )
    assert np.allclose(ps_calibration.kappa, kappa_true, rtol=1e-4)
    assert np.allclose(ps_calibration.Rd, rd_true, rtol=1e-4)
    assert np.allclose(ps_calibration.Rf, rd_true * kappa_true * 1e3, rtol=1e-4)
    assert np.allclose(ps_calibration.chi_squared_per_deg, 0)  # Noise free

    assert np.allclose(ps_calibration.err_fc, err_fc)
    assert np.allclose(ps_calibration.err_D, err_d)
    assert np.allclose(ps_calibration.err_f_diode, err_f_diode)
    assert np.allclose(ps_calibration.err_alpha, err_alpha)

    assert ps_calibration.is_success() is True

    # This field actually never gets set, which is very strange. But for exact parity, we keep it.
    setattr(ps_calibration, "error", "Error!")
    assert ps_calibration.is_success() is False


def test_bad_calibration_result_arg():
    with pytest.raises(TypeError):
        psc.CalibrationResults(bad_arg=5)


def test_bad_data():
    num_samples = 30000
    data = np.sin(.1*np.arange(num_samples))
    model = PassiveCalibrationModel(1, temperature=20, viscosity=0.0001)
    power_spectrum = psc.PowerSpectrum(data, num_samples)

    with pytest.raises(psc.CalibrationError):
        psc.fit_power_spectrum(power_spectrum, model=model)


def test_actual_spectrum():
    data = np.load(os.path.join(os.path.dirname(__file__), "reference_spectrum.npz"))
    reference_spectrum = data["arr_0"]
    model = PassiveCalibrationModel(4.4, temperature=20, viscosity=0.001002)
    reference_spectrum = psc.calculate_power_spectrum(reference_spectrum, sampling_rate=78125,
                                                      num_points_per_block=100, fit_range=(100.0, 23000.0))
    ps_calibration = psc.fit_power_spectrum(power_spectrum=reference_spectrum, model=model)

    assert np.allclose(ps_calibration.D, 0.0018512665210876748, rtol=1e-4)
    assert np.allclose(ps_calibration.Rd, 7.253645956145265, rtol=1e-4)
    assert np.allclose(ps_calibration.Rf, 1243.9711315478219, rtol=1e-4)
    assert np.allclose(ps_calibration.kappa, 0.17149598134079505, rtol=1e-4)
    assert np.allclose(ps_calibration.alpha, 0.5006103727942776, rtol=1e-4)
    assert np.allclose(ps_calibration.backing, 66.4331056392512, rtol=1e-4)
    assert np.allclose(ps_calibration.chi_squared_per_deg, 1.063783302378645, rtol=1e-4)
    assert np.allclose(ps_calibration.err_fc, 32.23007993226726, rtol=1e-4)
    assert np.allclose(ps_calibration.err_D, 6.43082000774291e-05, rtol=1e-4)
    assert np.allclose(ps_calibration.err_alpha, 0.013141463933316694, rtol=1e-4)
    assert np.allclose(ps_calibration.err_f_diode, 561.6377089699399, rtol=1e-4)
