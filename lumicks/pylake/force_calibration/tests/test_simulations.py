import pytest
import numpy as np
from lumicks.pylake.force_calibration.power_spectrum_calibration import calculate_power_spectrum
from .data.simulate_calibration_data import generate_active_calibration_test_data
from .data.simulate_ideal import simulate_calibration_data


@pytest.mark.slow
def test_simulation():
    """Compares two methods to simulate the power spectrum."""

    params = {
        "duration": 10,
        "sample_rate": 78125,
        "bead_diameter": 1.01,
        "stiffness": 0.4,
        "viscosity": 1.002e-3,
        "temperature": 20,
        "pos_response_um_volt": 0.618,
        "driving_sinusoid": (500, 31.95633),
        "diode": (0.4, 10000),
    }

    sim1_position, sim1_nanostage = simulate_calibration_data(
        **params, anti_aliasing=True, oversampling=16
    )
    sim2_position, sim2_nanostage = generate_active_calibration_test_data(**params)

    def power(data):
        return calculate_power_spectrum(data, params["sample_rate"], fit_range=(1, 2e4)).power

    # Check whether the spectra are close. Note that these tolerances are pretty loose, but the
    # errors quickly get very big.
    np.testing.assert_allclose(power(sim2_position) / power(sim1_position), 1, atol=2e-1)
    np.testing.assert_allclose(power(sim2_nanostage) / power(sim1_nanostage), 1, atol=2e-1)


def test_stability_throw():
    """Test whether simulation function throws when simulation would be unstabble"""
    with pytest.raises(RuntimeError):
        simulate_calibration_data(
            duration=1,
            sample_rate=78125,
            stiffness=1,
            bead_diameter=1,
            viscosity=1e-3,
            temperature=20,
            pos_response_um_volt=1,
            anti_aliasing=False,
            oversampling=1,
        )
