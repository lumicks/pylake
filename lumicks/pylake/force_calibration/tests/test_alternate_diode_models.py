import numpy as np
from lumicks.pylake.force_calibration.calibration_models import (
    FixedDiodeModel,
    PassiveCalibrationModel,
    ActiveCalibrationModel,
)
from lumicks.pylake.force_calibration.power_spectrum_calibration import (
    calculate_power_spectrum,
    fit_power_spectrum,
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


def test_fixed_diode(reference_models):
    """Test fixed diode model"""
    models, power_spectrum = model_and_data(reference_models, diode_alpha=0.4, fast_sensor=True)
    for model in models:
        model._filter = FixedDiodeModel(14000)
        fit = fit_power_spectrum(power_spectrum, model=model, bias_correction=False)

        np.testing.assert_allclose(fit.results["fc"].value, 4000, 1e-6)
        np.testing.assert_allclose(fit.results["D"].value, 1.14632, 1e-6)
        np.testing.assert_allclose(fit.results["alpha"].value, 0.4, 1e-6)

        # Diode frequency is a parameter now and not a result
        assert fit.params["f_diode"].value == 14000
        assert "f_diode" not in fit.results

        # Fix diode to the wrong frequency. We should not get a great fit that way.
        model._filter = FixedDiodeModel(13000)
        fit = fit_power_spectrum(power_spectrum, model=model, bias_correction=False)
        assert abs(fit.results["fc"].value - 4000) > 1.0
        assert abs(fit.results["D"].value - 1.14632) > 1e-2
