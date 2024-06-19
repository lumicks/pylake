import pytest

import lumicks.pylake as lk
from lumicks.pylake.force_calibration.validation import *
from lumicks.pylake.force_calibration.validation import upsample_frequency_axis
from lumicks.pylake.force_calibration.power_spectrum import PowerSpectrum
from lumicks.pylake.force_calibration.tests.data.simulate_calibration_data import (
    generate_active_calibration_test_data,
)


def simulate_test_data(bead_diameter=1.04, stiffness=0.1, diode=(0.52, 13000)):
    model_pars = {
        "bead_diameter": bead_diameter,
        "viscosity": 1.1e-3,
        "temperature": 25,
        "rho_sample": 997.0,
        "rho_bead": 1040.0,
        "distance_to_surface": None,
        "hydrodynamically_correct": True,
    }

    sim_pars = {
        "sample_rate": 78125,
        "stiffness": stiffness,
        "pos_response_um_volt": 0.618,
        "driving_sinusoid": (500, 17),
        "diode": diode,
    }

    volts, stage = generate_active_calibration_test_data(10, **sim_pars, **model_pars)

    return volts, stage, model_pars


@pytest.mark.slow
def test_hydro_problem():
    np.random.seed(1337)

    diameter = 1.6
    volts, _, model_pars = simulate_test_data(bead_diameter=diameter, stiffness=0.5)
    model_pars = model_pars | {
        "sample_rate": 78125,
        "num_points_per_block": 350,
    }
    no_hydro = {"hydrodynamically_correct": False}

    # In bulk and reasonably far from the surface, we should flag that we can do better!
    for distance in (None, 0.750001 * diameter):
        dist = {"distance_to_surface": distance}

        # With hydro we should be silent
        _ = lk.calibrate_force(volts, **(model_pars | dist))

        with pytest.raises(
            RuntimeError,
            match="the hydrodynamically correct model will lead to more accurate force calibrations",
        ):
            # Without we should issue an error
            fit = lk.calibrate_force(volts, **(model_pars | no_hydro | dist))
            check_hydro_enabled(fit)

    # Close to the surface, we shouldn't report it, since we don't support hydro there.
    fit = lk.calibrate_force(volts, **(model_pars | {"distance_to_surface": 0.75 * diameter}))
    check_hydro_enabled(fit)

    # Small beads we shouldn't report it, since hydro is not that relevant
    fit = lk.calibrate_force(volts, **(model_pars | {"bead_diameter": 1}))
    check_hydro_enabled(fit)


@pytest.mark.slow
def test_diode_problem():
    np.random.seed(1337)

    volts_ok, _, model_pars = simulate_test_data(stiffness=0.5)

    # Should be ok, since fc << f_diode
    fit = lk.calibrate_force(volts_ok, sample_rate=78125, num_points_per_block=350, **model_pars)
    check_diode_identifiability(fit, 0.1)

    # Should be also be ok, since fc << f_diode, despite f_diode having huge confidence intervals
    volts_ok, _, model_pars = simulate_test_data(stiffness=0.2, diode=(0.52, 40000))
    fit = lk.calibrate_force(volts_ok, sample_rate=78125, num_points_per_block=350, **model_pars)
    check_diode_identifiability(fit, 0.1)

    with pytest.raises(
        RuntimeError,
        match="estimate for the parasitic filtering frequency falls within the confidence interval "
        "for the corner frequency",
    ):
        # High stiffness to provoke the diode problem
        volts_bad, _, model_pars = simulate_test_data(stiffness=1.2)
        fit = lk.calibrate_force(
            volts_bad, sample_rate=78125, num_points_per_block=350, **model_pars
        )
        check_diode_identifiability(fit, 0.1)


@pytest.mark.slow
def test_fc_issues():
    np.random.seed(1337)

    volts, _, model_pars = simulate_test_data(stiffness=0.02)
    pars = model_pars | {"sample_rate": 78125, "num_points_per_block": 350}

    _ = lk.calibrate_force(volts, **pars, fit_range=(200, 23000))

    with pytest.raises(RuntimeError, match="Consider lowering the minimum fit range"):
        fit = lk.calibrate_force(volts, **pars, fit_range=(400, 23000))
        check_corner_frequency(fit)


@pytest.mark.slow
def test_backing_issues():
    np.random.seed(1337)

    volts, _, model_pars = simulate_test_data(stiffness=0.2)
    pars = model_pars | {"sample_rate": 78125, "num_points_per_block": 350}

    fit = lk.calibrate_force(volts, **pars)
    check_backing(fit, 1e-6)

    # corrupt the data with a noise spike
    volts += 4e-4 * np.sin(2.0 * np.pi * 13154 * np.arange(0.0, 10.0, 1.0 / 78125))

    with pytest.raises(
        RuntimeError, match="It is possible that the fit of the thermal calibration spectrum is bad"
    ):
        fit = lk.calibrate_force(volts, **pars, fit_range=(400, 23000))
        check_backing(fit, 1e-6)


@pytest.mark.slow
@pytest.mark.parametrize(
    "downsampling_factor, fit_range", [(66, None), (100, None), (51, None), (51, (100, 23000))]
)
def test_ps_upsampling(downsampling_factor, fit_range):
    data = np.random.normal(size=10000)
    ps_hf = PowerSpectrum(data, 78125)
    if fit_range:
        ps_hf = ps_hf.in_range(100, 23000)

    ps_ds = ps_hf.downsampled_by(downsampling_factor)

    last = downsampling_factor * (len(ps_hf.frequency) // downsampling_factor)
    np.testing.assert_allclose(upsample_frequency_axis(ps_ds).flatten(), ps_hf.frequency[:last])


@pytest.mark.slow
def test_fc_inaccuracy():
    np.random.seed(1337)

    # high stiffness leads to uncertainty in fc
    volts, _, model_pars = simulate_test_data(stiffness=1.5)
    pars = model_pars | {"sample_rate": 78125, "num_points_per_block": 350}

    fit = lk.calibrate_force(volts, **pars)
    with pytest.raises(RuntimeError, match="More than 20% error in the corner frequency"):
        check_calibration_factor_precision(fit)
