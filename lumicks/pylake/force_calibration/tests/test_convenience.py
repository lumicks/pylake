import pytest
import numpy as np
from lumicks.pylake.force_calibration.convenience import calibrate_force
from lumicks.pylake.force_calibration.calibration_models import (
    PassiveCalibrationModel,
    ActiveCalibrationModel,
)


def test_passive_force_calibration_active(reference_models):
    """This test tests merely whether values were handed to the low level API correctly, not whether
    the calibration results are correct."""
    data, f_sample = reference_models.lorentzian_td(4000, 1, 0.4, 14000, 78125)
    fit = calibrate_force(
        data,
        2,
        9,
        viscosity=6,
        active_calibration=False,
        hydrodynamically_correct=True,
        rho_sample=600.0,
        rho_bead=1000.0,
        distance_to_surface=16,
        fast_sensor=True,
        fit_range=(1e1, 8e3),
        num_points_per_block=64,
        excluded_ranges=[[0, 100]],
        drag=42,
    )
    params = {
        "Bead diameter": 2,
        "Temperature": 9,
        "Viscosity": 6,
        "Sample density": 600.0,
        "Bead density": 1000.0,
        "Distance to surface": 16,
        "Points per block": 64,
    }
    for key, value in params.items():
        assert fit.params[key].value == value

    assert isinstance(fit.model, PassiveCalibrationModel)
    assert "f_diode" not in fit.results  # fast sensor
    assert fit.results["gamma_ex_lateral"].value == 42  # verify the drag has been carried
    assert fit.ps_data.frequency.min() > 100  # verify that the exclusion range was passed
    assert fit.ps_data.frequency.max() < 8e3  # verify that the bounds were passed


def test_active_force_calibration_active(reference_models):
    """This test tests merely whether values were handed to the low level API correctly, not whether
    the calibration results are correct."""
    data, f_sample = reference_models.lorentzian_td(4000, 1, 0.4, 14000, 78125)
    fit = calibrate_force(
        data,
        2,
        9,
        driving_data=np.sin(77 * 2 * np.pi * np.arange(data.size) / 78125),
        driving_frequency_guess=77,
        viscosity=6,
        active_calibration=True,
        hydrodynamically_correct=True,
        rho_sample=600.0,
        rho_bead=1000.0,
        distance_to_surface=16,
        fast_sensor=True,
        fit_range=(1e1, 8e3),
        num_points_per_block=64,
        excluded_ranges=[[0, 100]],
    )
    params = {
        "Bead diameter": 2,
        "Temperature": 9,
        "Viscosity": 6,
        "Sample density": 600.0,
        "Bead density": 1000.0,
        "Distance to surface": 16,
        "Points per block": 64,
        "Driving frequency (guess)": 77,
    }
    for key, value in params.items():
        assert fit.params[key].value == value

    assert isinstance(fit.model, ActiveCalibrationModel)
    assert "f_diode" not in fit.results  # fast sensor
    assert fit.ps_data.frequency.min() > 100  # verify that the exclusion range was passed
    assert fit.ps_data.frequency.max() < 8e3  # verify that the bounds were passed


def test_invalid_options_calibration():
    with pytest.raises(ValueError, match="Active calibration is not supported for axial force"):
        calibrate_force([1], 1, 20, axial=True, active_calibration=True)

    with pytest.raises(
        ValueError, match="Drag coefficient cannot be carried over to active calibration"
    ):
        calibrate_force([1], 1, 20, drag=5, active_calibration=True)

    with pytest.raises(
        ValueError, match="When using fast_sensor=True, there is no diode model to fix"
    ):
        calibrate_force([1], 1, 20, fast_sensor=True, fixed_diode=150)

    with pytest.raises(
        ValueError, match="Active calibration requires the driving_data to be defined"
    ):
        calibrate_force([1], 1, 20, active_calibration=True)


def test_mandatory_keyworded_arguments():
    with pytest.raises(TypeError, match="takes 3 positional arguments but 5 were given"):
        calibrate_force([], 1, 2, 3, 4)
