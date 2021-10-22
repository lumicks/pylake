import pytest
import numpy as np
from lumicks.pylake.force_calibration.power_spectrum_calibration import (
    calculate_power_spectrum,
    fit_power_spectrum,
)
from lumicks.pylake.force_calibration.calibration_models import (
    PassiveCalibrationModel,
    ActiveCalibrationModel,
)
from lumicks.pylake.force_calibration.detail.power_models import g_diode
from lumicks.pylake.force_calibration.detail.hydrodynamics import *
from .data.simulate_calibration_data import generate_active_calibration_test_data


@pytest.mark.parametrize(
    "gamma0,bead_radius,rho_bead,ref_frequency",
    [
        [1.16e-8, 0.52e-6, 1060.0, 2957151.776732485741],
        [1.26e-8, 0.52e-6, 1050.0, 3242669.879313553683],
        [2.16e-8, 1.04e-6, 1100.0, 663273.384405044955],
        [4.06e-8, 2.2e-6, 1060.0, 136673.2897802220250],
        [8.19e-8, 4.4e-6, 1060.0, 34462.87694889219710],
    ],
)
def test_calculate_dissipation_frequency(gamma0, bead_radius, rho_bead, ref_frequency):
    np.testing.assert_allclose(
        calculate_dissipation_frequency(gamma0, bead_radius, rho_bead), ref_frequency
    )


@pytest.mark.parametrize(
    "gamma0,bead_radius,rho_sample,height,real_ref,imag_ref",
    [
        [
            1.16e-8,
            0.52e-6,
            997.0,
            25e-9,
            [2.146433851928, 2.096663632939, 2.024434965031, 1.992226321743, 1.974849294243],
            [0.011630458607, 0.05320652377, 0.097554582022, 0.11210831639, 0.118719431369],
        ],
        [
            1.16e-8,
            0.52e-6,
            1030.0,
            100e-9,
            [1.88376122823, 1.844894022809, 1.788826929015, 1.764035562532, 1.750731200458],
            [0.009066095721, 0.040893020824, 0.073061104239, 0.082745785304, 0.086845661542],
        ],
        [
            1.26e-8,
            0.52e-6,
            997.0,
            100e-9,
            [1.88428545869, 1.847547960417, 1.794253959758, 1.770518194101, 1.757726945946],
            [0.008574402811, 0.038989785254, 0.070582165064, 0.080471057325, 0.084785171374],
        ],
        [
            2.16e-8,
            1.04e-6,
            997.0,
            100e-9,
            [2.031589268351, 1.941110176159, 1.825379221619, 1.780993280624, 1.75895180847],
            [0.021100970348, 0.083264945626, 0.120277352273, 0.122416869131, 0.120672716538],
        ],
        [
            4.06e-8,
            2.2e-6,
            997.0,
            100e-9,
            [2.109035647269, 1.909177848289, 1.732121982481, 1.687183568937, 1.669628797901],
            [0.048702354672, 0.137826558553, 0.112174672075, 0.077069222082, 0.054969073856],
        ],
        [
            8.19e-8,
            4.4e-6,
            997.0,
            100e-9,
            [2.107549029073, 1.790854573074, 1.658981842986, 1.660709835682, 1.671593540813],
            [0.088996455997, 0.136432997301, -0.019548698139, -0.100441746115, -0.144369823553],
        ],
    ],
)
def test_complex_drag(gamma0, rho_sample, bead_radius, height, real_ref, imag_ref):
    f = np.array([37.0, 1000.0, 5000.0, 8000.0, 10000.0])
    real, imag = calculate_complex_drag(f, gamma0, rho_sample, bead_radius, bead_radius + height)
    np.testing.assert_allclose(real, real_ref, rtol=1e-6)
    np.testing.assert_allclose(imag, imag_ref, rtol=1e-6)


@pytest.mark.parametrize(
    "gamma0,bead_radius,rho_sample",
    [
        [1.16e-8, 0.52e-6, 997.0],
        [1.16e-8, 0.52e-6, 1030.0],
        [1.26e-8, 0.52e-6, 997.0],
        [2.16e-8, 1.04e-6, 997.0],
        [4.06e-8, 2.2e-6, 997.0],
        [8.19e-8, 4.4e-6, 997.0],
    ],
)
def test_complex_drag_limit_case(gamma0, rho_sample, bead_radius):
    # Compare whether the model for far away from the surface results in the same model as the
    # complex model using a large height.
    f = np.array([37.0, 1000.0, 5000.0, 8000.0, 10000.0])
    real, imag = calculate_complex_drag(f, gamma0, rho_sample, bead_radius, 10)  # 10 meters away
    real_far, imag_far = calculate_complex_drag(f, gamma0, rho_sample, bead_radius, None)
    np.testing.assert_allclose(real, real_far, rtol=1e-6)
    np.testing.assert_allclose(imag, imag_far, rtol=1e-6)


@pytest.mark.parametrize(
    "fc,diffusion,f_diode,alpha,gamma0,bead_radius,rho_sample,rho_bead,height,ref_power_spectrum",
    [
        [
            4800,
            1.08,
            14000,
            0.4,
            1.16e-8,
            0.52e-6,
            997.0,
            1060.0,
            25e-9,
            [1.018966397e-08, 8.174707470e-09, 1.538233969e-09, 6.041701478e-10, 3.637960655e-10],
        ],
        [
            4800,
            1.08,
            14000,
            0.4,
            1.16e-8,
            0.52e-6,
            1030.0,
            1060.0,
            100e-9,
            [8.943606396e-09, 7.491080241e-09, 1.661923005e-09, 6.693646549e-10, 4.056422851e-10],
        ],
        [
            4800,
            1.08,
            14000,
            0.4,
            1.26e-8,
            0.52e-6,
            997.0,
            1060.0,
            100e-9,
            [8.946161914e-09, 7.504194956e-09, 1.661270710e-09, 6.680646956e-10, 4.045687481e-10],
        ],
        [
            2445,
            0.542,
            14000,
            0.4,
            1.978234e-8,
            1.04e-6,
            997.0,
            1060.0,
            100e-9,
            [1.862354180e-08, 1.043633041e-08, 9.856086009e-10, 3.640796920e-10, 2.161805755e-10],
        ],
        [
            1165,
            0.25,
            14000,
            0.4,
            4.2694e-8,
            2.2e-6,
            997.0,
            1060.0,
            100e-9,
            [3.909336166e-08, 9.039080438e-09, 5.111435690e-10, 1.837124500e-10, 1.079757337e-10],
        ],
        [
            545.57,
            0.1266,
            14000,
            0.4,
            7.557e-8,
            4.4e-6,
            997.0,
            1060.0,
            100e-9,
            [8.776759369e-08, 6.334322297e-09, 2.794904018e-10, 9.255996313e-11, 5.152615472e-11],
        ],
    ],
)
def test_hydro_spectra(
    fc,
    diffusion,
    f_diode,
    alpha,
    gamma0,
    bead_radius,
    rho_sample,
    rho_bead,
    height,
    ref_power_spectrum,
):
    f = np.array([37.0, 1000.0, 5000.0, 8000.0, 10000.0])
    power_spectrum = (
        passive_power_spectrum_model_hydro(
            f,
            fc=fc,
            diffusion_constant=diffusion,
            gamma0=gamma0,
            bead_radius=bead_radius,
            rho_sample=rho_sample,
            rho_bead=rho_bead,
            distance_to_surface=bead_radius + height,
        )
        * g_diode(f, f_diode, alpha)
    )

    np.testing.assert_allclose(power_spectrum, ref_power_spectrum)


@pytest.mark.parametrize(
    "amp,fc,gamma0,bead_radius,rho_sample,rho_bead,height,ref_power",
    [
        [500e-6, 4800, 1.16e-8, 0.52e-6, 997.0, 1060.0, 25e-9, 3.420441139e-11],
        [250e-6, 4800, 1.16e-8, 0.52e-6, 1030.0, 1060.0, 100e-9, 6.586892852e-12],
        [500e-6, 4800, 1.26e-8, 0.52e-6, 997.0, 1060.0, 100e-9, 2.636236948e-11],
        [500e-6, 2445, 1.978234e-8, 1.04e-6, 997.0, 1060.0, 100e-9, 1.178556597e-10],
        [500e-6, 1165, 4.2694e-8, 2.2e-6, 997.0, 1060.0, 100e-9, 5.576540852e-10],
        [500e-6, 545.57, 7.557e-8, 4.4e-6, 997.0, 1060.0, 100e-9, 2.467139096e-09],
    ],
)
def test_hydro_power(
    amp,
    fc,
    gamma0,
    bead_radius,
    rho_sample,
    rho_bead,
    height,
    ref_power,
):
    power = theoretical_driving_power_hydrodynamics(
        driving_frequency=37,
        driving_amplitude=amp,
        fc=fc,
        gamma0=gamma0,
        bead_radius=bead_radius,
        rho_sample=rho_sample,
        rho_bead=rho_bead,
        distance_to_surface=bead_radius + height,
    )

    np.testing.assert_allclose(power, ref_power)


def test_integration_active_calibration_hydrodynamics(integration_test_parameters):
    shared_pars, simulation_pars = integration_test_parameters

    np.random.seed(10071985)
    volts, nanostage = generate_active_calibration_test_data(10, **simulation_pars, **shared_pars)
    model = ActiveCalibrationModel(
        nanostage,
        volts,
        **shared_pars,
        sample_rate=simulation_pars["sample_rate"],
        driving_frequency_guess=33,
    )
    power_spectrum = calculate_power_spectrum(volts, simulation_pars["sample_rate"])
    fit = fit_power_spectrum(power_spectrum, model, bias_correction=False)
    expected_params = {
        "Sample density": 997.0,
        "Bead density": 1040.0,
        "Distance to surface": 0.7776500000000001,
        "Bead diameter": 1.03,
        "Viscosity": 0.0011,
        "Temperature": 25,
        "Driving frequency (guess)": 33,
        "Sample rate": 78125,
        "num_windows": 5,
        "Max iterations": 10000,
        "Fit tolerance": 1e-07,
        "Points per block": 2000,
    }
    expected_results = {
        "Rd": 0.6092796748780891,
        "kappa": 0.10388246375443001,
        "Rf": 63.29347374183399,
        "gamma_0": 1.0678273429551705e-08,
        "gamma_ex": 1.0989730336350438e-08,
        "fc": 1504.4416105821158,
        "D": 1.0090151317063,
        "err_fc": 13.075876724291339,
        "err_D": 0.0066021439072302835,
        "f_diode": 14675.638696737586,
        "alpha": 0.41651098052983593,
        "err_f_diode": 352.2917702189488,
        "err_alpha": 0.014231238753589254,
        "chi_squared_per_deg": 0.8659867914094764,
        "backing": 14.340689726784328,
    }

    for key, value in expected_params.items():
        np.testing.assert_allclose(fit.params[key].value, value, err_msg=key)

    for key, value in expected_results.items():
        np.testing.assert_allclose(fit.results[key].value, value, err_msg=key)


def test_integration_passive_calibration_hydrodynamics(integration_test_parameters):
    shared_pars, simulation_pars = integration_test_parameters

    np.random.seed(10071985)
    volts, _ = generate_active_calibration_test_data(10, **simulation_pars, **shared_pars)
    model = PassiveCalibrationModel(**shared_pars)
    power_spectrum = calculate_power_spectrum(volts, simulation_pars["sample_rate"])
    fit = fit_power_spectrum(power_spectrum, model, bias_correction=False)

    expected_params = {
        "Sample density": 997.0,
        "Bead density": 1040.0,
        "Distance to surface": 0.7776500000000001,
        "Bead diameter": 1.03,
        "Viscosity": 0.0011,
        "Temperature": 25,
        "Max iterations": 10000,
        "Fit tolerance": 1e-07,
        "Points per block": 2000,
        "Sample rate": 78125,
    }
    expected_results = {
        "Rd": 0.6181013468813382,
        "kappa": 0.10093835959160387,
        "Rf": 62.39013601556319,
        "gamma_0": 1.0678273429551705e-08,
        "fc": 1504.4416105821158,
        "D": 1.0090151317063,
        "err_fc": 13.075876724291339,
        "err_D": 0.0066021439072302835,
        "f_diode": 14675.638696737586,
        "alpha": 0.41651098052983593,
        "err_f_diode": 352.2917702189488,
        "err_alpha": 0.014231238753589254,
        "chi_squared_per_deg": 0.8659867914094764,
        "backing": 14.340689726784328,
    }

    for key, value in expected_params.items():
        np.testing.assert_allclose(fit.params[key].value, value, err_msg=key)

    for key, value in expected_results.items():
        np.testing.assert_allclose(fit.results[key].value, value, err_msg=key)


def test_integration_active_calibration_hydrodynamics_bulk(integration_test_parameters):
    shared_pars, simulation_pars = integration_test_parameters

    np.random.seed(10071985)
    shared_pars["distance_to_surface"] = None
    volts, nanostage = generate_active_calibration_test_data(10, **simulation_pars, **shared_pars)
    model = ActiveCalibrationModel(
        nanostage,
        volts,
        **shared_pars,
        sample_rate=simulation_pars["sample_rate"],
        driving_frequency_guess=33,
    )
    power_spectrum = calculate_power_spectrum(volts, simulation_pars["sample_rate"])
    fit = fit_power_spectrum(power_spectrum, model, bias_correction=False)

    expected_params = {
        "Sample density": 997.0,
        "Bead density": 1040.0,
        "Bead diameter": 1.03,
        "Viscosity": 0.0011,
        "Temperature": 25,
        "Driving frequency (guess)": 33,
        "Sample rate": 78125,
        "num_windows": 5,
        "Max iterations": 10000,
        "Fit tolerance": 1e-07,
        "Points per block": 2000,
    }
    expected_results = {
        "Rd": 0.6095674943889238,
        "kappa": 0.10359295685924054,
        "Rf": 63.14689914902713,
        "gamma_0": 1.0678273429551705e-08,
        "gamma_ex": 1.0978355408018856e-08,
        "fc": 1501.803370440244,
        "D": 1.0091069801313286,
        "f_diode": 14669.862556235465,
        "alpha": 0.41657472149713015,
        "err_fc": 11.599562805624199,
        "err_D": 0.007332334985757522,
        "err_f_diode": 376.8360414675165,
        "err_alpha": 0.014653541838852356,
        "chi_squared_per_deg": 0.8692145118092963,
        "backing": 14.917612794899505,
    }

    assert fit.params["Distance to surface"].value is None
    for key, value in expected_params.items():
        np.testing.assert_allclose(fit.params[key].value, value, err_msg=key)

    for key, value in expected_results.items():
        np.testing.assert_allclose(fit.results[key].value, value, err_msg=key)


def test_distance_to_surface_input(integration_test_parameters):
    signal = np.cos(2 * np.pi * 37 * np.arange(0, 1, 1.0 / 78125))
    pars = {
        "bead_diameter": 1.0,
        "driving_data": signal,
        "force_voltage_data": signal,
        "driving_frequency_guess": 37,
        "sample_rate": 78125,
    }

    with pytest.raises(
        ValueError, match="Distance from bead center to surface is smaller than the bead radius"
    ):
        ActiveCalibrationModel(distance_to_surface=0.49, hydrodynamically_correct=True, **pars)

    with pytest.raises(
        ValueError,
        match="This model is only valid for distances to the surface larger "
        "than 1.5 times the bead radius. Distances closer to the surface "
        "are currently not supported.",
    ):
        ActiveCalibrationModel(distance_to_surface=0.51, hydrodynamically_correct=True, **pars)

    # Distance passes the check
    ActiveCalibrationModel(distance_to_surface=0.5 * 1.51, hydrodynamically_correct=True, **pars)

    # Passes because we're not using the distance in this case.
    ActiveCalibrationModel(distance_to_surface=None, hydrodynamically_correct=True, **pars)

    with pytest.raises(NotImplementedError):
        ActiveCalibrationModel(
            distance_to_surface=0.5 * 1.51, hydrodynamically_correct=False, **pars
        )
