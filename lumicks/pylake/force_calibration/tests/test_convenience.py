import pickle

import numpy as np
import pytest

from lumicks.pylake.force_calibration.convenience import calibrate_force
from lumicks.pylake.force_calibration.calibration_models import (
    ActiveCalibrationModel,
    PassiveCalibrationModel,
)


def test_passive_force_calibration_active(reference_models):
    """This test tests merely whether values were handed to the low level API correctly, not whether
    the calibration results are correct."""
    data, f_sample = reference_models.lorentzian_td(4000, 1, 0.4, 14000, 78125)
    fit = calibrate_force(
        data,
        2,
        9,
        sample_rate=78125,
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
    oscillation = np.sin(77 * 2 * np.pi * np.arange(data.size) / 78125)

    # We need a driving peak in the power spectrum, otherwise we obtain a negative value for the
    # peak power which leads to a downstream warning.
    data += oscillation

    fit = calibrate_force(
        data,
        2,
        9,
        sample_rate=78125,
        driving_data=oscillation,
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
    common_args = {"active_calibration": True, "sample_rate": 78125}
    with pytest.raises(ValueError, match="Active calibration is not supported for axial force"):
        calibrate_force([1], 1, 20, axial=True, **common_args)

    with pytest.raises(
        ValueError, match="Drag coefficient cannot be carried over to active calibration"
    ):
        calibrate_force([1], 1, 20, drag=5, **common_args)

    with pytest.raises(
        ValueError, match="When using fast_sensor=True, there is no diode model to fix"
    ):
        calibrate_force([1], 1, 20, fast_sensor=True, fixed_diode=150, sample_rate=78125)

    with pytest.raises(
        ValueError, match="When using fast_sensor=True, there is no diode model to fix"
    ):
        calibrate_force([1], 1, 20, fast_sensor=True, fixed_alpha=0.4, sample_rate=78125)

    with pytest.raises(
        ValueError, match="Active calibration requires the driving_data to be defined"
    ):
        calibrate_force([1], 1, 20, **common_args)

    with pytest.raises(
        ValueError, match="Approximate driving frequency must be specified and larger than zero"
    ):
        calibrate_force([1], 1, 20, **common_args, driving_data=np.array([1]))

    for driving_freq in (0, -1):
        with pytest.raises(
            ValueError, match="Approximate driving frequency must be specified and larger than zero"
        ):
            calibrate_force(
                [1],
                1,
                20,
                **common_args,
                driving_data=np.array([1]),
                driving_frequency_guess=driving_freq,
            )


def test_mandatory_keyworded_arguments():
    with pytest.raises(TypeError, match="takes 3 positional arguments but 5 were given"):
        calibrate_force([], 1, 2, 3, 4)


def test_diode_fixing(reference_models):
    data, f_sample = reference_models.lorentzian_td(4000, 1, 0.4, 14000, 78125)
    fit = calibrate_force(data, 1, 20, fixed_diode=1000, sample_rate=78125)
    assert "f_diode" in fit.params
    assert "alpha" not in fit.params

    fit = calibrate_force(data, 1, 20, fixed_alpha=0.5, sample_rate=78125)
    assert "f_diode" not in fit.params
    assert "alpha" in fit.params

    fit = calibrate_force(data, 1, 20, fixed_diode=14000, fixed_alpha=0.5, sample_rate=78125)
    assert "f_diode" in fit.params
    assert "alpha" in fit.params

    fit = calibrate_force(data, 1, 20, fixed_alpha=0, sample_rate=78125)
    assert "f_diode" not in fit.params
    assert "alpha" in fit.params

    with pytest.raises(ValueError, match="Fixed diode frequency must be larger than zero."):
        calibrate_force(data, 1, 20, fixed_diode=0, sample_rate=78125)


def test_pickling(tmpdir_factory, reference_models):
    data, f_sample = reference_models.lorentzian_td(4000, 1, 0.4, 14000, 78125)
    fit = calibrate_force(data, 1, 20, sample_rate=78125)

    tmpdir = tmpdir_factory.mktemp("pylake")
    tmppath = f"{tmpdir}/test.pkl"
    with open(tmppath, "wb") as f:
        pickle.dump(fit, f)

    # Verify that we can open the model again
    with open(tmppath, "rb") as f:
        pickled_fit = pickle.load(f)

    np.testing.assert_allclose(pickled_fit["kappa"].value, 0.12173556860362873)
