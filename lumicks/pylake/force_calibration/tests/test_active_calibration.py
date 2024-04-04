import numpy as np
import scipy
import pytest

from lumicks.pylake.force_calibration.calibration_models import ActiveCalibrationModel
from lumicks.pylake.force_calibration.detail.drag_models import faxen_factor
from lumicks.pylake.force_calibration.detail.power_models import sphere_friction_coefficient
from lumicks.pylake.force_calibration.power_spectrum_calibration import (
    fit_power_spectrum,
    calculate_power_spectrum,
)

from .data.simulate_calibration_data import generate_active_calibration_test_data


@pytest.mark.parametrize(
    "stiffness, viscosity, temperature, pos_response_um_volt, driving_sinusoid, diode, driving_"
    "frequency_guess",
    [
        [0.1, 1.002e-3, 20, 0.618, (500, 31.95633), (0.4, 15000), 32],
        [0.2, 1.012e-3, 20, 1.618, (500, 31.95633), (0.4, 14000), 32],
        [0.3, 1.002e-3, 50, 1.618, (300, 30.42633), (0.4, 16000), 29],
    ],
)
def test_integration_active_calibration(
    compare_to_reference_dict,
    stiffness,
    viscosity,
    temperature,
    pos_response_um_volt,
    driving_sinusoid,
    diode,
    driving_frequency_guess,
):
    import scipy.constants

    """Functional end to end test for active calibration"""
    sample_rate, bead_diameter = 78125, 1.03
    np.random.seed(0)
    force_voltage_data, driving_data = generate_active_calibration_test_data(
        duration=20,
        sample_rate=sample_rate,
        bead_diameter=bead_diameter,
        stiffness=stiffness,
        viscosity=viscosity,
        temperature=temperature,
        pos_response_um_volt=pos_response_um_volt,
        driving_sinusoid=driving_sinusoid,
        diode=diode,
    )

    model = ActiveCalibrationModel(
        driving_data,
        force_voltage_data,
        sample_rate,
        bead_diameter,
        driving_frequency_guess,
        viscosity,
        temperature,
    )

    # Validate estimation of the driving input
    np.testing.assert_allclose(model.driving_amplitude, driving_sinusoid[0] * 1e-9, rtol=1e-5)
    np.testing.assert_allclose(model.driving_frequency, driving_sinusoid[1], rtol=1e-5)

    power_spectrum = calculate_power_spectrum(force_voltage_data, sample_rate)
    fit = fit_power_spectrum(power_spectrum, model)

    np.testing.assert_allclose(fit["kappa"].value, stiffness, rtol=5e-2)
    np.testing.assert_allclose(fit["alpha"].value, diode[0], rtol=5e-2)
    np.testing.assert_allclose(fit["f_diode"].value, diode[1], rtol=5e-2)
    np.testing.assert_allclose(fit["Rd"].value, pos_response_um_volt, rtol=5e-2)

    response_calc = fit["Rd"].value * fit["kappa"].value * 1e3
    np.testing.assert_allclose(fit["Rf"].value, response_calc, rtol=1e-9)

    kt = scipy.constants.k * scipy.constants.convert_temperature(temperature, "C", "K")
    drag_coeff_calc = kt / (fit["D"].value * fit["Rd"].value ** 2)
    np.testing.assert_allclose(
        fit["gamma_0"].value,
        sphere_friction_coefficient(viscosity, bead_diameter * 1e-6),
        rtol=1e-9,
    )
    np.testing.assert_allclose(fit["gamma_ex"].value, drag_coeff_calc * 1e12, rtol=1e-9)
    np.testing.assert_allclose(
        fit["local_drag_coefficient"].value, drag_coeff_calc * 1e12, rtol=1e-9
    )

    np.testing.assert_allclose(fit["Bead diameter"].value, bead_diameter)
    np.testing.assert_allclose(fit["Driving frequency (guess)"].value, driving_frequency_guess)
    np.testing.assert_allclose(fit["Sample rate"].value, sample_rate)
    np.testing.assert_allclose(fit["Viscosity"].value, viscosity)
    np.testing.assert_allclose(fit["num_windows"].value, 5)

    np.testing.assert_allclose(
        fit["driving_amplitude"].value, driving_sinusoid[0] * 1e-3, rtol=1e-5
    )
    np.testing.assert_allclose(fit["driving_frequency"].value, driving_sinusoid[1], rtol=1e-5)

    compare_to_reference_dict(
        {
            par: fit[par].value
            for par in (
                "driving_power",
                "err_driving_power",
                "err_theoretical_power",
                "theoretical_power",
                "err_kappa",
                "err_Rd",
            )
        }
    )


def test_bias_correction():
    """Functional end to end test for active calibration"""

    np.random.seed(0)
    force_voltage_data, driving_data = generate_active_calibration_test_data(
        duration=20,
        sample_rate=78125,
        bead_diameter=1.03,
        stiffness=0.2,
        viscosity=1.002e-3,
        temperature=20,
        pos_response_um_volt=0.618,
        driving_sinusoid=(500, 31.95633),
        diode=(0.4, 13000),
    )

    model = ActiveCalibrationModel(driving_data, force_voltage_data, 78125, 1.03, 32, 1.002e-3, 20)

    # Low blocking deliberately leads to higher bias (so it's easier to measure)
    block_size = 3
    power_spectrum_low = calculate_power_spectrum(
        force_voltage_data, 78125, num_points_per_block=block_size
    )

    fit_biased = fit_power_spectrum(power_spectrum_low, model, bias_correction=False)
    fit_debiased = fit_power_spectrum(power_spectrum_low, model, bias_correction=True)

    bias_corr = block_size / (block_size + 1)
    np.testing.assert_allclose(fit_debiased["D"].value, fit_biased["D"].value * bias_corr)
    np.testing.assert_allclose(fit_debiased["err_D"].value, fit_biased["err_D"].value * bias_corr)

    # Biased vs debiased estimates (in comments are the reference values for N_pts_per_block = 150
    # Note how the estimates are better on the right.
    comparisons = {
        "fc": [3310.651532245893, 3310.651532245893],  # Ref: 3277.6576037747836
        "D": [1.472922058628551, 1.1046915439714131],  # Ref: 1.0896306365192108
        "kappa": [0.15317517466591019, 0.2043759281959786],  # Ref: 0.20106518840690035
        "Rd": [0.6108705452113169, 0.6106577513480039],  # Ref: 0.6168083172053238
        "Rf": [93.57020246100325, 124.8037447418174],  # Ref: 124.0186805098316
    }

    for key, values in comparisons.items():
        for fit, value in zip([fit_biased, fit_debiased], values):
            np.testing.assert_allclose(fit[key].value, value)

    assert fit_biased.params["Bias correction"].value is False
    assert fit_debiased.params["Bias correction"].value is True


def test_faxen_correction_active(active_calibration_surface_data):
    """Active calibration should barely be affected by surface corrections for the drag coefficient.
    However, the interpretation of gamma_ex, which may be carried over to the other calibration
    *is* important, so this should be covered by a specific test."""
    shared_pars, sim_pars, active_pars = active_calibration_surface_data
    power_spectrum = calculate_power_spectrum(
        active_pars["force_voltage_data"], sim_pars["sample_rate"]
    )

    model = ActiveCalibrationModel(**active_pars, **shared_pars, hydrodynamically_correct=False)
    fit = fit_power_spectrum(power_spectrum, model, bias_correction=False)

    # Fitting with *no* hydrodynamically correct model, but *with* Faxen's law
    np.testing.assert_allclose(fit.results["Rd"].value, 0.5979577465734786)
    np.testing.assert_allclose(fit.results["kappa"].value, 0.10852140970454485)
    np.testing.assert_allclose(fit.results["Rf"].value, 64.89121760190687)
    # gamma_0 and gamma_ex should be the same, since gamma_ex is corrected to be "in bulk".
    np.testing.assert_allclose(fit.results["gamma_0"].value, 1.0678273429551705e-08)
    np.testing.assert_allclose(fit.results["gamma_ex"].value, 1.1271667835127709e-08)
    np.testing.assert_allclose(
        fit.results["local_drag_coefficient"].value,
        1.1271667835127709e-08
        * faxen_factor(shared_pars["distance_to_surface"], shared_pars["bead_diameter"] / 2),
    )

    # Disabling Faxen's correction on the drag makes the estimates *much* worse
    model = ActiveCalibrationModel(
        **active_pars, **dict(shared_pars, distance_to_surface=None), hydrodynamically_correct=False
    )
    fit = fit_power_spectrum(power_spectrum, model, bias_correction=False)

    np.testing.assert_allclose(fit.results["Rd"].value, 0.5979577465734786)
    np.testing.assert_allclose(fit.results["kappa"].value, 0.10852140970454485)
    np.testing.assert_allclose(fit.results["Rf"].value, 64.89121760190687)
    # Not affected since this is gamma bulk
    np.testing.assert_allclose(fit.results["gamma_0"].value, 1.0678273429551705e-08)
    # The drag is now much different, since we're not using Faxen's law to back-correct the drag
    # to its actual bulk value.
    np.testing.assert_allclose(
        fit.results[model._measured_drag_fieldname].value, 1.571688034506783e-08
    )
    # The local estimate is now the same.
    np.testing.assert_allclose(fit.results["local_drag_coefficient"].value, 1.571688034506783e-08)


def test_hydro_active(active_calibration_surface_data):
    shared_pars, sim_pars, active_pars = active_calibration_surface_data
    power_spectrum = calculate_power_spectrum(
        active_pars["force_voltage_data"], sim_pars["sample_rate"]
    )
    model = ActiveCalibrationModel(**active_pars, **shared_pars, hydrodynamically_correct=True)
    fit = fit_power_spectrum(power_spectrum, model, bias_correction=False)

    np.testing.assert_allclose(fit.results["Rd"].value, 0.6093861540574103)
    np.testing.assert_allclose(fit.results["gamma_0"].value, 1.0678273429551705e-08)
    # Note proximity to gamma_0
    gamma_ex = fit.results["gamma_ex"].value
    np.testing.assert_allclose(gamma_ex, 1.0984573910537512e-08)
    np.testing.assert_allclose(fit.results["local_drag_coefficient"].value, 1.537177770849885e-08)
    np.testing.assert_allclose(
        fit.results["local_drag_coefficient"].value,
        gamma_ex
        / (1 - (9 / 16) * 0.5 * shared_pars["bead_diameter"] / shared_pars["distance_to_surface"]),
    )

    model = ActiveCalibrationModel(
        **active_pars, **dict(shared_pars, distance_to_surface=None), hydrodynamically_correct=True
    )
    fit = fit_power_spectrum(power_spectrum, model, bias_correction=False)
    # No way to correct back to bulk without a height
    np.testing.assert_allclose(fit.results["gamma_ex"].value, 1.6258510187033216e-08)
    np.testing.assert_allclose(fit.results["local_drag_coefficient"].value, 1.6258510187033216e-08)
