import numpy as np
import pytest

from lumicks.pylake.force_calibration.calibration_models import (
    ActiveCalibrationModel,
    PassiveCalibrationModel,
)
from lumicks.pylake.force_calibration.detail.power_models import (
    motion_blur_peak,
    motion_blur_spectrum,
    passive_power_spectrum_model,
    theoretical_driving_power_lorentzian,
)
from lumicks.pylake.force_calibration.power_spectrum_calibration import (
    fit_power_spectrum,
    calculate_power_spectrum,
)

from .data.simulate_ideal import simulate_calibration_data


def test_motion_blur_spectrum():
    acquisition_time = 1.0 / 3.0
    f = np.arange(100, 23000, 1000)
    passive_model = PassiveCalibrationModel(bead_diameter=1, fast_sensor=True)
    passive_model_blurred = passive_model._motion_blur(acquisition_time)
    assert id(passive_model) != id(passive_model_blurred)

    osc = np.sin(2.0 * np.pi * 2.0 / 200.0 * np.arange(1000))
    active_model = ActiveCalibrationModel(
        driving_data=osc,
        force_voltage_data=osc,
        driving_frequency_guess=2.0,
        sample_rate=100,
        bead_diameter=1,
        fast_sensor=True,
    )
    active_model_blurred = active_model._motion_blur(acquisition_time)
    assert id(active_model) != id(active_model_blurred)

    for model in (passive_model_blurred, active_model_blurred):
        np.testing.assert_allclose(
            model(f, fc=100, diffusion_constant=0.1),
            motion_blur_spectrum(passive_power_spectrum_model, acquisition_time=acquisition_time)(
                f, fc=100, diffusion_constant=0.1
            ),
        )


def test_motion_blur_peak():
    acquisition_time = 1.0 / 3.0
    osc = np.sin(2.0 * np.pi * 2.0 / 200.0 * np.arange(1000))
    model = ActiveCalibrationModel(
        driving_data=osc,
        force_voltage_data=osc,
        driving_frequency_guess=2.0,
        sample_rate=100,
        bead_diameter=1,
        fast_sensor=True,
    )._motion_blur(acquisition_time)

    np.testing.assert_allclose(
        model._theoretical_driving_power_model(100),
        motion_blur_peak(
            theoretical_driving_power_lorentzian,
            driving_frequency=model.driving_frequency,
            acquisition_time=acquisition_time,
        )(
            fc=100,
            driving_frequency=model.driving_frequency,
            driving_amplitude=model.driving_amplitude,
        ),
    )


@pytest.mark.slow
def test_camera_calibration():
    np.random.seed(909)
    f_drive = 16.8
    sample_rate = 500
    shared_params = {
        "bead_diameter": 1.01,
        "viscosity": 1.002e-3,
        "temperature": 20,
    }
    params = {
        **shared_params,
        "duration": 600,
        "sample_rate": sample_rate,
        "stiffness": 0.05,
        "pos_response_um_volt": 0.618,
        "driving_sinusoid": (500, f_drive),
        "diode": None,
    }

    oversampling = 128
    pos, nano = simulate_calibration_data(
        **params, anti_aliasing="integrate", oversampling=oversampling
    )
    passive_model = PassiveCalibrationModel(**shared_params, fast_sensor=True)

    active_model = ActiveCalibrationModel(
        force_voltage_data=pos,
        driving_data=nano,
        **shared_params,
        sample_rate=sample_rate,
        driving_frequency_guess=f_drive,
        fast_sensor=True,
    )

    # Consider aliasing and motion blur
    models = (
        model._motion_blur(1.0 / sample_rate)._alias_model(sample_rate, 20)
        for model in (passive_model, active_model)
    )

    distance_responses = [0.6234771740528027, 0.623431094357452]
    stiffnesses = [0.04950015364112712, 0.04950747132648579]
    force_responses = [30.86221590734949, 30.864497027941212]

    ps = calculate_power_spectrum(pos, sample_rate, fit_range=(25, 1000), num_points_per_block=200)
    for model, rd, kappa, rf in zip(models, distance_responses, stiffnesses, force_responses):
        calibration = fit_power_spectrum(ps, model)
        np.testing.assert_allclose(calibration["Rd"].value, rd)
        np.testing.assert_allclose(calibration["Rf"].value, rf)
        np.testing.assert_allclose(calibration["kappa"].value, kappa)
