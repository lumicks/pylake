import numpy as np
import scipy.constants
import pytest
from lumicks.pylake.force_calibration.power_spectrum_calibration import (
    calculate_power_spectrum,
    fit_power_spectrum,
)
from lumicks.pylake.force_calibration.calibration_models import ActiveCalibrationModel
from .data.simulate_calibration_data import generate_active_calibration_test_data


@pytest.mark.parametrize(
    "sample_rate, bead_diameter, stiffness, viscosity, temperature, pos_response_um_volt, "
    "driving_sinusoid, diode, driving_frequency_guess, power_density",
    [
        [78125, 1.03, 0.1, 1.002e-3, 20, 0.618, (500, 31.95633), (0.4, 15000), 32, 1.958068e-5],
        [78125, 1.03, 0.2, 1.012e-3, 20, 1.618, (500, 31.95633), (0.4, 14000), 32, 7.28664e-07],
        [78125, 1.03, 0.3, 1.002e-3, 50, 1.618, (300, 30.42633), (0.4, 16000), 29, 1.098337e-07],
    ],
)
def test_integration_active_calibration(
    sample_rate,
    bead_diameter,
    stiffness,
    viscosity,
    temperature,
    pos_response_um_volt,
    driving_sinusoid,
    diode,
    driving_frequency_guess,
    power_density
):
    """Functional end to end test for active calibration"""

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

    np.testing.assert_allclose(model._response_power_density, power_density, rtol=1e-5)
    num_points_per_window = int(np.round(sample_rate * model.num_windows / model.driving_frequency))
    freq_axis = np.fft.rfftfreq(num_points_per_window, 1.0 / sample_rate)
    np.testing.assert_allclose(model._frequency_bin_width, freq_axis[1] - freq_axis[0])

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
    np.testing.assert_allclose(fit["gamma_0"].value, drag_coeff_calc * 1e12, rtol=1e-9)

    np.testing.assert_allclose(fit["Bead diameter"].value, bead_diameter)
    np.testing.assert_allclose(fit["Driving frequency (guess)"].value, driving_frequency_guess)
    np.testing.assert_allclose(fit["Sample rate"].value, sample_rate)
    np.testing.assert_allclose(fit["Viscosity"].value, viscosity)
    np.testing.assert_allclose(fit["num_windows"].value, 5)
