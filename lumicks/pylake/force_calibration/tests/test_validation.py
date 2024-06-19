from typing import Union, Optional
from dataclasses import dataclass

import pytest

from lumicks.pylake.force_calibration.validation import *
from lumicks.pylake.force_calibration.calibration_models import PassiveCalibrationModel


@dataclass
class MockParameter:
    description: str
    value: Union[float, int, bool]
    unit: str


@dataclass
class MockPowerSpectrum:
    frequency: np.ndarray
    power: np.ndarray
    num_points_per_block: int

    @property
    def frequency_bin_width(self):
        # Assumes a 10-second spectrum
        return self.num_points_per_block / 10


@dataclass
class MockCalibration:
    model: Optional[PassiveCalibrationModel] = None
    ps_model: Optional[MockPowerSpectrum] = None
    ps_data: Optional[MockPowerSpectrum] = None
    params: Optional[dict] = None
    results: Optional[dict] = None
    fitted_params: Optional[np.ndarray] = None

    def __call__(self, frequency):
        """Evaluate the spectral model for one or more frequencies

        Parameters
        ----------
        frequency : array_like
            One or more frequencies at which to evaluate the spectral model.
        """
        return self.model(frequency, *self.fitted_params)


def test_diode_issue():
    # Fine calibration (stiffness = 0.5, fc << f_diode)
    results = {
        "fc": MockParameter("corner frequency", 7622.98, "Hz"),
        "err_fc": MockParameter("corner frequency std err", 340.813, "Hz"),
        "f_diode": MockParameter("corner frequency std err", 12182.2, "Hz"),
        "err_f_diode": MockParameter("corner frequency std err", 1261.42, "Hz"),
    }
    cal = MockCalibration(results=results)
    assert not check_diode_identifiability(cal, 0.1)

    # Fine calibration (stiffness = 0.2, fc << f_diode)
    results = {
        "fc": MockParameter("corner frequency", 2965.87, "Hz"),
        "err_fc": MockParameter("corner frequency std err", 20.4309, "Hz"),
        "f_diode": MockParameter("corner frequency std err", 38171.4, "Hz"),
        "err_f_diode": MockParameter("corner frequency std err", 11850.8, "Hz"),
    }
    cal = MockCalibration(results=results)
    assert not check_diode_identifiability(cal, 0.1)

    # Problematic calibration (stiffness = 1.2)
    results = {
        "fc": MockParameter("corner frequency", 16788.5, "Hz"),
        "err_fc": MockParameter("corner frequency std err", 3041.78, "Hz"),
        "f_diode": MockParameter("corner frequency std err", 13538.5, "Hz"),
        "err_f_diode": MockParameter("corner frequency std err", 1280.43, "Hz"),
    }
    cal = MockCalibration(results=results)
    warning = check_diode_identifiability(cal, 0.1)
    assert (
        "estimate for the parasitic filtering frequency falls within the confidence interval "
        "for the corner frequency"
    ) in warning


def test_blocking_issue():
    def gen_calibration(fc, num_points_per_block):
        model = PassiveCalibrationModel(bead_diameter=1.0)
        freq = np.arange(150, 22000, 0.1 * num_points_per_block)
        return MockCalibration(
            model=model,
            fitted_params=np.array([fc, 1.0, 0.5, 14000]),
            ps_model=MockPowerSpectrum(freq, freq, num_points_per_block=num_points_per_block),
        )

    # The error in stiffness is expected to be proportional to 1/fc and the number of points per
    # block. We first test the over-blocked case (> 5% stiffness error).
    warning = check_blocking_error(gen_calibration(fc=200, num_points_per_block=2000), 3)
    assert "Maximum spectral error exceeds threshold" in warning

    # Fine
    assert not check_blocking_error(gen_calibration(fc=200, num_points_per_block=500), 3)

    # Also fine
    assert not check_blocking_error(gen_calibration(fc=1000, num_points_per_block=2000), 3)


def test_hydro():
    def generate_calibration(bead_diameter, distance, hydro):
        params = {
            "Distance to surface": MockParameter("distance to surface", distance, "um"),
            "Bead diameter": MockParameter("bead diameter", bead_diameter, "um"),
        }
        model = PassiveCalibrationModel(
            bead_diameter=bead_diameter,
            distance_to_surface=distance,
            hydrodynamically_correct=hydro,
        )
        return MockCalibration(params=params, model=model)

    # Small beads are fine either way
    assert not check_hydro_enabled(
        generate_calibration(bead_diameter=1.0, distance=None, hydro=False)
    )
    assert not check_hydro_enabled(
        generate_calibration(bead_diameter=1.0, distance=None, hydro=True)
    )

    # Very close to the surface, we shouldn't flag since we don't support hydro so close
    assert not check_hydro_enabled(
        generate_calibration(bead_diameter=4.0, distance=0.75 * 4.0, hydro=False)
    )

    # In bulk and reasonably far from the surface, we should flag that we can do better if
    # hydro is off!
    diameter = 4.0
    for current_distance in (None, 0.750001 * diameter):
        # With hydro we should be silent
        shared_pars = {"bead_diameter": diameter, "distance": current_distance}
        assert not check_hydro_enabled(generate_calibration(**shared_pars, hydro=True))

        warning = check_hydro_enabled(generate_calibration(**shared_pars, hydro=False))
        assert (
            "the hydrodynamically correct model will lead to more accurate force calibrations"
            in warning
        )


def test_corner_frequency_too_low():
    def gen_calibration(fc, min_freq, num_points_per_block):
        freq = np.arange(min_freq, 22000, 0.1 * num_points_per_block)
        return MockCalibration(
            results={
                "fc": MockParameter("corner frequency", fc, "Hz"),
            },
            ps_data=MockPowerSpectrum(freq, freq, num_points_per_block=num_points_per_block),
        )

    assert not check_corner_frequency(gen_calibration(250, 200, 100))

    warning = check_corner_frequency(gen_calibration(150, 200, 100))
    assert (
        "Estimated corner frequency (150 Hz) is below lowest frequency in the power spectrum "
        "(200 Hz)"
    ) in warning

    assert not check_corner_frequency(gen_calibration(200, 200, 2000))

    warning = check_corner_frequency(gen_calibration(200, 200, 2001))
    assert "Corner frequency (200 Hz) is below the spectral frequency resolution" in warning


def test_goodness_of_fit_check():
    cal = MockCalibration(results={"backing": MockParameter("statistical backing", 1e-6, "%")})
    assert not check_backing(cal, 0.99e-6)

    warning = check_backing(cal, 1.01e-6)
    assert "Statistical backing is low (1.000e-06 < 1.010e-06)" in warning


@pytest.mark.parametrize(
    "fc_factor, rd_factor, kappa_factor, message",
    [
        (0.21, 0.19, 0.19, "More than 20% error in the corner frequency"),
        (0.19, 0.21, 0.19, "More than 20% error in the displacement sensitivity"),
        (0.19, 0.19, 0.21, "More than 20% error in the stiffness"),
    ],
)
def test_high_uncertainty(fc_factor, rd_factor, kappa_factor, message):
    fc, rd, kappa = 7622.98, 0.647278, 1.04987
    results = {
        "fc": MockParameter("corner frequency", fc, "Hz"),
        "err_fc": MockParameter("corner frequency std err", fc_factor * fc, "Hz"),
        "Rd": MockParameter("Displacement sensitivity", rd, "um/V"),
        "err_Rd": MockParameter("Displacement sensitivity std err", rd_factor * rd, "um/V"),
        "kappa": MockParameter("Stiffness", kappa, "um/V"),
        "err_kappa": MockParameter("Stiffness std err", kappa_factor * kappa, "um/V"),
    }

    warning = check_calibration_factor_precision(MockCalibration(results=results))
    assert message in warning


def test_good_calibration():
    diameter = 4.81
    err_factor = 0.1  # std err factor of 0.1 is acceptable
    num_points_per_block = 200
    fc, rd, kappa = 7622.98, 0.647278, 1.04987

    results = {
        "fc": MockParameter("corner frequency", fc, "Hz"),
        "err_fc": MockParameter("corner frequency std err", err_factor * fc, "Hz"),
        "Rd": MockParameter("Displacement sensitivity", rd, "um/V"),
        "err_Rd": MockParameter("Displacement sensitivity std err", err_factor * rd, "um/V"),
        "kappa": MockParameter("Stiffness", kappa, "um/V"),
        "err_kappa": MockParameter("Stiffness std err", err_factor * kappa, "um/V"),
        "backing": MockParameter("statistical backing", 44.43523, "%"),
    }
    params = {
        "Distance to surface": MockParameter("distance to surface", 50.0, "um"),
        "Bead diameter": MockParameter("bead diameter", diameter, "um"),
    }
    model = PassiveCalibrationModel(bead_diameter=diameter, hydrodynamically_correct=True)
    freq = np.arange(150, 22000, 0.1 * num_points_per_block)
    mock_ps = MockPowerSpectrum(freq, freq, num_points_per_block=num_points_per_block)
    cal = MockCalibration(
        model=model,
        fitted_params=np.array([4000, 1.0, 0.5, 14000]),
        ps_model=mock_ps,
        ps_data=mock_ps,
        params=params,
        results=results,
    )

    assert not validate_results(cal, alpha=0.1, desired_backing=1, blocking_threshold=1)

    warning = validate_results(cal, alpha=0.1, desired_backing=50, blocking_threshold=1)
    assert "Statistical backing is low (4.444e+01 < 5.000e+01)" in warning
