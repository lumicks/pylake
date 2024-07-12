from dataclasses import dataclass

import numpy as np
import pytest

from lumicks.pylake.force_calibration.power_spectrum import PowerSpectrum
from lumicks.pylake.force_calibration.detail.validation import (
    check_backing,
    validate_results,
    check_hydro_enabled,
    check_blocking_error,
    check_corner_frequency,
    compare_to_robust_fitting,
    check_diode_identifiability,
    check_calibration_factor_precision,
)
from lumicks.pylake.force_calibration.calibration_models import PassiveCalibrationModel
from lumicks.pylake.force_calibration.power_spectrum_calibration import (
    CalibrationResults,
    CalibrationParameter,
    fit_power_spectrum,
)


def generate_spectrum(frequency, power, num_points_per_block=200):
    """Creates a PowerSpectrum directly by generating a tiny one, and overwriting its contents.
    Note: this should be replaced with an actual simpler constructor on the Pylake side."""
    ps = PowerSpectrum(np.ones(10), 78125)
    ps.frequency = frequency
    ps.power = power
    ps.total_duration = 10
    ps.total_sampled_used = 78125 * 10
    ps.num_points_per_block = num_points_per_block
    return ps


def make_calibration_result(
    model=None, ps_model=None, ps_data=None, params=None, results=None, fitted_params=None
):
    # These are mandatory entries whose existence gets checked upon initialization.
    mandatory_results = {
        "kappa": CalibrationParameter("stiffness", 1.0, "pN/nm"),
        "Rf": CalibrationParameter("force sensitivity", 1.0, "pN/V"),
    }

    return CalibrationResults(
        model,
        ps_model,
        ps_data,
        params,
        mandatory_results if results is None else mandatory_results | results,
        fitted_params,
    )


def test_diode_issue():
    # Fine calibration (stiffness = 0.5, fc << f_diode)
    results = {
        "fc": CalibrationParameter("corner frequency", 7622.98, "Hz"),
        "err_fc": CalibrationParameter("corner frequency std err", 340.813, "Hz"),
        "f_diode": CalibrationParameter("corner frequency std err", 12182.2, "Hz"),
        "err_f_diode": CalibrationParameter("corner frequency std err", 1261.42, "Hz"),
    }
    cal = make_calibration_result(results=results)
    assert not check_diode_identifiability(cal, 0.1)

    # Fine calibration (stiffness = 0.2, fc << f_diode)
    results = {
        "fc": CalibrationParameter("corner frequency", 2965.87, "Hz"),
        "err_fc": CalibrationParameter("corner frequency std err", 20.4309, "Hz"),
        "f_diode": CalibrationParameter("corner frequency std err", 38171.4, "Hz"),
        "err_f_diode": CalibrationParameter("corner frequency std err", 11850.8, "Hz"),
    }
    cal = make_calibration_result(results=results)
    assert not check_diode_identifiability(cal, 0.1)

    # Problematic calibration (stiffness = 1.2)
    results = {
        "fc": CalibrationParameter("corner frequency", 16788.5, "Hz"),
        "err_fc": CalibrationParameter("corner frequency std err", 3041.78, "Hz"),
        "f_diode": CalibrationParameter("corner frequency std err", 13538.5, "Hz"),
        "err_f_diode": CalibrationParameter("corner frequency std err", 1280.43, "Hz"),
    }
    cal = make_calibration_result(results=results)
    warning = check_diode_identifiability(cal, 0.1)
    assert (
        "estimate for the parasitic filtering frequency falls within the confidence interval "
        "for the corner frequency"
    ) in warning


def test_blocking_issue():
    def gen_calibration(fc, num_points_per_block):
        model = PassiveCalibrationModel(bead_diameter=1.0)
        freq = np.arange(150, 22000, 0.1 * num_points_per_block)
        return make_calibration_result(
            model=model,
            fitted_params=np.array([fc, 1.0, 0.5, 14000]),
            ps_model=generate_spectrum(freq, freq, num_points_per_block=num_points_per_block),
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
    def generate_calibration(bead_diameter, distance, hydro, axial=False):
        params = {
            "Distance to surface": CalibrationParameter("distance to surface", distance, "um"),
            "Bead diameter": CalibrationParameter("bead diameter", bead_diameter, "um"),
        }
        model = PassiveCalibrationModel(
            bead_diameter=bead_diameter,
            distance_to_surface=distance,
            hydrodynamically_correct=hydro,
            axial=axial,
        )
        return make_calibration_result(params=params, model=model)

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

    # We don't support hydro for axial at the moment, so don't flag it
    assert not check_hydro_enabled(
        generate_calibration(bead_diameter=4.0, hydro=False, distance=None, axial=True)
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
        return make_calibration_result(
            results={"fc": CalibrationParameter("corner frequency", fc, "Hz")},
            ps_data=generate_spectrum(freq, freq, num_points_per_block=num_points_per_block),
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
    cal = make_calibration_result(
        results={"backing": CalibrationParameter("statistical backing", 1e-6, "%")}
    )
    assert not check_backing(cal, 0.99e-6)

    warning = check_backing(cal, 1.01e-6)
    assert "Statistical backing is low (1.000e-06 < 1.010e-06)" in warning


@pytest.mark.parametrize(
    "fc_factor, rd_factor, kappa_factor, factor, message",
    [
        (0.21, 0.19, 0.19, 0.2, "More than 20% error in the corner frequency"),
        (0.19, 0.21, 0.19, 0.2, "More than 20% error in the displacement sensitivity"),
        (0.19, 0.19, 0.21, 0.2, "More than 20% error in the stiffness"),
        (0.14, 0.14, 0.16, 0.15, "More than 15% error in the stiffness"),
    ],
)
def test_high_uncertainty(fc_factor, rd_factor, kappa_factor, factor, message):
    fc, rd, kappa = 7622.98, 0.647278, 1.04987
    results = {
        "fc": CalibrationParameter("corner frequency", fc, "Hz"),
        "err_fc": CalibrationParameter("corner frequency std err", fc_factor * fc, "Hz"),
        "Rd": CalibrationParameter("Displacement sensitivity", rd, "um/V"),
        "err_Rd": CalibrationParameter("Displacement sensitivity std err", rd_factor * rd, "um/V"),
        "kappa": CalibrationParameter("Stiffness", kappa, "um/V"),
        "err_kappa": CalibrationParameter("Stiffness std err", kappa_factor * kappa, "um/V"),
    }

    warning = check_calibration_factor_precision(
        make_calibration_result(results=results), factor=factor
    )
    assert message in warning


def test_robust_comparison():
    fc, diffusion_constant, diameter, num_points_per_block = 1945.9546, 0.0274, 2.1, 200

    model = PassiveCalibrationModel(
        bead_diameter=diameter, hydrodynamically_correct=True, temperature=25
    )
    true_params = np.array([fc, diffusion_constant, 0.446, 14312])
    freq = np.arange(150, 22000, 0.1 * num_points_per_block)
    sim = model(freq, *true_params)
    mock_ps = generate_spectrum(freq, sim, num_points_per_block=num_points_per_block)

    mock_ps.power[200] *= 5  # This noise spike leads to an error of 13% in kappa
    calibration = fit_power_spectrum(mock_ps, model)
    assert compare_to_robust_fitting(calibration, 0.14) is None

    assert compare_to_robust_fitting(calibration, 0.13) == (
        "More than 13% difference between robust and regular fit in parameter estimates:\n"
        + "kappa: regular: 0.248, robust: 0.215, rel_diff: 13%\n"
    )

    assert compare_to_robust_fitting(calibration, 0.11) == (
        "More than 11% difference between robust and regular fit in parameter estimates:\n"
        + "kappa: regular: 0.248, robust: 0.215, rel_diff: 13%\n"
        "Rd: regular: 0.000182, robust: 0.000204, rel_diff: 12%\n"
    )

    assert compare_to_robust_fitting(calibration, 0.02) == (
        "More than 2% difference between robust and regular fit in parameter estimates:\n"
        + "Rf: regular: 0.045, robust: 0.0439, rel_diff: 2%\n"
        "kappa: regular: 0.248, robust: 0.215, rel_diff: 13%\n"
        "Rd: regular: 0.000182, robust: 0.000204, rel_diff: 12%\n"
    )


def test_good_calibration():
    fc, diffusion_constant, diameter, num_points_per_block = 1945.9546, 0.0274, 2.1, 200

    model = PassiveCalibrationModel(
        bead_diameter=diameter, hydrodynamically_correct=True, temperature=25
    )
    true_params = np.array([fc, diffusion_constant, 0.446, 14312])
    freq = np.arange(150, 22000, 0.1 * num_points_per_block)
    sim = model(freq, *true_params)
    mock_ps = generate_spectrum(freq, sim, num_points_per_block=num_points_per_block)
    cal = fit_power_spectrum(mock_ps, model)

    assert not validate_results(cal, alpha=0.1, desired_backing=1, blocking_threshold=1)

    mock_ps.power[500::4] *= 0.75  # mangle the spectrum
    cal = fit_power_spectrum(mock_ps, model)
    warning = validate_results(cal, alpha=0.1, desired_backing=50, blocking_threshold=1)
    assert "Statistical backing is low (1.047e-19 < 5.000e+01)" in warning
