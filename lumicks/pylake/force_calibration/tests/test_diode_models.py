import re
from copy import deepcopy

import numpy as np
import pytest

from lumicks.pylake.force_calibration.calibration_models import (
    FixedDiodeModel,
    ActiveCalibrationModel,
    PassiveCalibrationModel,
    diode_params_from_voltage,
)
from lumicks.pylake.force_calibration.power_spectrum_calibration import (
    fit_power_spectrum,
    calculate_power_spectrum,
)


def model_and_data(reference_models, diode_alpha, fast_sensor):
    """Generate model and data for this particular test"""
    params = {"bead_diameter": 1.03, "temperature": 20, "viscosity": 1.002e-3}
    data, f_sample = reference_models.lorentzian_td(
        4000, 1.14632, alpha=diode_alpha, f_diode=14000, num_samples=78125
    )
    passive_model = PassiveCalibrationModel(**params, fast_sensor=fast_sensor)
    sine = np.sin(32.0 * 2.0 * np.pi * np.arange(0, 1, 1.0 / f_sample))
    active_model = ActiveCalibrationModel(
        sine, sine, f_sample, driving_frequency_guess=32, **params, fast_sensor=fast_sensor
    )
    power_spectrum = calculate_power_spectrum(data, f_sample, num_points_per_block=5)
    return (passive_model, active_model), power_spectrum


def test_fit_fixed_pars(reference_models):
    """Test model without diode effect"""
    models, power_spectrum = model_and_data(reference_models, diode_alpha=1.0, fast_sensor=True)
    for model in models:
        fit = fit_power_spectrum(power_spectrum, model=model, bias_correction=False)
        np.testing.assert_allclose(fit.results["fc"].value, 4000, 1e-6)
        np.testing.assert_allclose(fit.results["D"].value, 1.14632, 1e-6)
        assert "f_diode" not in fit.results
        assert "alpha" not in fit.results


def test_underfit_fast_sensor(reference_models):
    """Diode effect in the data, but the model doesn't actually have it. Should have a poor fit."""
    models, power_spectrum = model_and_data(reference_models, diode_alpha=0.4, fast_sensor=True)
    for model in models:
        fit = fit_power_spectrum(power_spectrum, model=model)
        assert abs(fit.results["fc"].value - 4000) > 1.0
        assert abs(fit.results["D"].value - 1.14632) > 0.1


@pytest.mark.parametrize(
    "model_params, fixed_params, free_params",
    [
        [{"diode_frequency": 14000}, {"f_diode": 14000}, {"alpha": 0.4}],
        [{"diode_alpha": 0.4}, {"alpha": 0.4}, {"f_diode": 14000}],
        [{"diode_frequency": 14000, "diode_alpha": 0.4}, {"f_diode": 14000, "alpha": 0.4}, {}],
    ],
)
def test_fixed_f_diode(model_params, fixed_params, free_params, reference_models):
    """Test fixed diode model"""
    models, power_spectrum = model_and_data(reference_models, diode_alpha=0.4, fast_sensor=False)
    for model in models:
        model._filter = FixedDiodeModel(**model_params)
        fit = fit_power_spectrum(power_spectrum, model=model, bias_correction=False)
        np.testing.assert_allclose(fit(fit.ps_model.frequency), fit.ps_model.power)
        ref_params = [fit[p].value for p in ("fc", "D", "f_diode", "alpha") if p in fit.results]
        np.testing.assert_allclose(fit.fitted_params, ref_params)

        # Test good fit
        np.testing.assert_allclose(fit.results["fc"].value, 4000, 1e-6)
        np.testing.assert_allclose(fit.results["D"].value, 1.14632, 1e-6)
        for key, value in free_params.items():
            np.testing.assert_allclose(fit.results[key].value, value, 1e-6)

        # Check whether parameters are actually parameters and not results
        for key, value in fixed_params.items():
            np.testing.assert_allclose(fit.params[key].value, value, 1e-6)
        for key in fixed_params.keys():
            assert key not in fit.results

        # Fix to the wrong value (this should mess up the fit)
        bad_params = deepcopy(model_params)
        for key in bad_params:
            bad_params[key] *= 1.1
        model._filter = FixedDiodeModel(**bad_params)
        fit = fit_power_spectrum(power_spectrum, model=model, bias_correction=False)
        np.testing.assert_allclose(fit(fit.ps_model.frequency), fit.ps_model.power)
        ref_params = [fit[p].value for p in ("fc", "D", "f_diode", "alpha") if p in fit.results]
        np.testing.assert_allclose(fit.fitted_params, ref_params)
        assert abs(fit.results["fc"].value - 4000) > 1.0
        assert abs(fit.results["D"].value - 1.14632) > 1e-2


def test_alpha_validity():
    # Alpha should be between 0 and 1
    for good_alpha in (0.0, 0.5, 1.0):
        FixedDiodeModel(diode_alpha=good_alpha)

    for bad_alpha in (1.000001, -0.000001):
        with pytest.raises(
            ValueError,
            match=re.escape("Diode relaxation factor should be between 0 and 1 (inclusive)."),
        ):
            FixedDiodeModel(diode_alpha=bad_alpha)


def test_power_model():
    """Tests the model used to look up diode parameters"""
    f_diode_coeffs = [4.33262833e03, 1.23960400e00, 1.41995232e04]
    alpha_coeffs = [0.05406556, 0.90222653, 0.56367068]

    trap_voltages = np.arange(-0.75, 2.6, 0.75)
    f_diodes = [9866.89490862, 9866.89490862, 12489.56087698, 13524.65084027, 13933.1707376]
    alphas = [0.50960512, 0.50960512, 0.53618879, 0.54970144, 0.55657002]

    for trap_voltage, ref_f_diode, ref_alpha in zip(trap_voltages, f_diodes, alphas):
        f_diode, alpha, power = diode_params_from_voltage(
            np.sin(0.2 * np.pi * np.arange(10)) + trap_voltage, *f_diode_coeffs, *alpha_coeffs
        )
        np.testing.assert_allclose(f_diode, ref_f_diode)
        np.testing.assert_allclose(alpha, ref_alpha)
        np.testing.assert_allclose(power, trap_voltage if trap_voltage > 0 else 0, atol=1e-15)
