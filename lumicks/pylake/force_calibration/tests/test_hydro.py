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
    power_spectrum = passive_power_spectrum_model_hydro(
        f,
        fc=fc,
        diffusion_constant=diffusion,
        f_diode=f_diode,
        alpha=alpha,
        gamma0=gamma0,
        bead_radius=bead_radius,
        rho_sample=rho_sample,
        rho_bead=rho_bead,
        distance_to_surface=bead_radius + height,
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
    fit = fit_power_spectrum(power_spectrum, model)

    np.testing.assert_allclose(fit.params["Sample density"].value, 997.0)
    np.testing.assert_allclose(fit.params["Bead density"].value, 1040.0)
    np.testing.assert_allclose(fit.params["Distance to surface"].value, 0.555)
    np.testing.assert_allclose(fit.params["Bead diameter"].value, 1.03)
    np.testing.assert_allclose(fit.params["Viscosity"].value, 0.0011)
    np.testing.assert_allclose(fit.params["Temperature"].value, 25)
    np.testing.assert_allclose(fit.params["Driving frequency (guess)"].value, 33)
    np.testing.assert_allclose(fit.params["Sample rate"].value, 78125)
    np.testing.assert_allclose(fit.params["num_windows"].value, 5)
    np.testing.assert_allclose(fit.params["Max iterations"].value, 10000)
    np.testing.assert_allclose(fit.params["Fit tolerance"].value, 1e-07)
    np.testing.assert_allclose(fit.params["Points per block"].value, 2000)

    np.testing.assert_allclose(fit.results["Rd"].value, 0.6051649330570094)
    np.testing.assert_allclose(fit.results["kappa"].value, 0.10543256286211536)
    np.testing.assert_allclose(fit.results["Rf"].value, 63.80408984648098)
    np.testing.assert_allclose(fit.results["gamma_0"].value, 1.0678273429551705e-08)
    np.testing.assert_allclose(fit.results["gamma_ex"].value, 1.1141608095967259e-08)
    np.testing.assert_allclose(fit.results["fc"].value, 1506.0764476562647)
    np.testing.assert_allclose(fit.results["D"].value, 1.0088409661297066)
    np.testing.assert_allclose(fit.results["f_diode"].value, 14678.932801906305)
    np.testing.assert_allclose(fit.results["alpha"].value, 0.41654289354446566)
    np.testing.assert_allclose(fit.results["err_fc"].value, 14.276233861052459, rtol=1e-4)
    np.testing.assert_allclose(fit.results["err_D"].value, 0.006159823623889375, rtol=1e-4)
    np.testing.assert_allclose(fit.results["err_f_diode"].value, 340.24103273566794, rtol=1e-4)
    np.testing.assert_allclose(fit.results["err_alpha"].value, 0.014047418012337998, rtol=1e-4)
    np.testing.assert_allclose(fit.results["chi_squared_per_deg"].value, 0.8650977340589278)
    np.testing.assert_allclose(fit.results["backing"].value, 14.184338818615572)


def test_integration_passive_calibration_hydrodynamics(integration_test_parameters):
    shared_pars, simulation_pars = integration_test_parameters

    np.random.seed(10071985)
    volts, _ = generate_active_calibration_test_data(10, **simulation_pars, **shared_pars)
    model = PassiveCalibrationModel(**shared_pars)
    power_spectrum = calculate_power_spectrum(volts, simulation_pars["sample_rate"])
    fit = fit_power_spectrum(power_spectrum, model)

    np.testing.assert_allclose(fit.params["Sample density"].value, 997.0)
    np.testing.assert_allclose(fit.params["Bead density"].value, 1040.0)
    np.testing.assert_allclose(fit.params["Distance to surface"].value, 0.555)
    np.testing.assert_allclose(fit.params["Bead diameter"].value, 1.03)
    np.testing.assert_allclose(fit.params["Viscosity"].value, 0.0011)
    np.testing.assert_allclose(fit.params["Temperature"].value, 25)
    np.testing.assert_allclose(fit.params["Max iterations"].value, 10000)
    np.testing.assert_allclose(fit.params["Fit tolerance"].value, 1e-07)
    np.testing.assert_allclose(fit.params["Points per block"].value, 2000)
    np.testing.assert_allclose(fit.params["Sample rate"].value, 78125)

    np.testing.assert_allclose(fit.results["Rd"].value, 0.6181546988640827)
    np.testing.assert_allclose(fit.results["kappa"].value, 0.10104804664845167)
    np.testing.assert_allclose(fit.results["Rf"].value, 62.46332484677742)
    np.testing.assert_allclose(fit.results["gamma_0"].value, 1.0678273429551705e-08)
    np.testing.assert_allclose(fit.results["fc"].value, 1506.0764476562647)
    np.testing.assert_allclose(fit.results["D"].value, 1.0088409661297066)
    np.testing.assert_allclose(fit.results["f_diode"].value, 14678.932801906305)
    np.testing.assert_allclose(fit.results["alpha"].value, 0.41654289354446566)
    np.testing.assert_allclose(fit.results["err_fc"].value, 14.276233861052459, rtol=1e-4)
    np.testing.assert_allclose(fit.results["err_D"].value, 0.006159823623889375, rtol=1e-4)
    np.testing.assert_allclose(fit.results["err_f_diode"].value, 340.24103273566794, rtol=1e-4)
    np.testing.assert_allclose(fit.results["err_alpha"].value, 0.014047418012337998, rtol=1e-4)
    np.testing.assert_allclose(fit.results["chi_squared_per_deg"].value, 0.8650977340589278)
    np.testing.assert_allclose(fit.results["backing"].value, 14.184338818615572)


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
    fit = fit_power_spectrum(power_spectrum, model)

    np.testing.assert_allclose(fit.params["Sample density"].value, 997.0)
    np.testing.assert_allclose(fit.params["Bead density"].value, 1040.0)
    assert fit.params["Distance to surface"].value is None
    np.testing.assert_allclose(fit.params["Bead diameter"].value, 1.03)
    np.testing.assert_allclose(fit.params["Viscosity"].value, 0.0011)
    np.testing.assert_allclose(fit.params["Temperature"].value, 25)
    np.testing.assert_allclose(fit.params["Driving frequency (guess)"].value, 33)
    np.testing.assert_allclose(fit.params["Sample rate"].value, 78125)
    np.testing.assert_allclose(fit.params["num_windows"].value, 5)
    np.testing.assert_allclose(fit.params["Max iterations"].value, 10000)
    np.testing.assert_allclose(fit.params["Fit tolerance"].value, 1e-07)
    np.testing.assert_allclose(fit.params["Points per block"].value, 2000)

    np.testing.assert_allclose(fit.results["Rd"].value, 0.6124727648954287)
    np.testing.assert_allclose(fit.results["kappa"].value, 0.10261253656600618)
    np.testing.assert_allclose(fit.results["Rf"].value, 62.84738398351507)
    np.testing.assert_allclose(fit.results["gamma_0"].value, 1.0678273429551705e-08)
    np.testing.assert_allclose(fit.results["gamma_ex"].value, 1.0874447128195706e-08)
    np.testing.assert_allclose(fit.results["fc"].value, 1501.8043883199641)
    np.testing.assert_allclose(fit.results["D"].value, 1.0091072985710965)
    np.testing.assert_allclose(fit.results["f_diode"].value, 14669.854380073006)
    np.testing.assert_allclose(fit.results["alpha"].value, 0.416574766034984)
    np.testing.assert_allclose(fit.results["err_fc"].value, 11.599581832014788, rtol=1e-4)
    np.testing.assert_allclose(fit.results["err_D"].value, 0.0073323528722267824, rtol=1e-4)
    np.testing.assert_allclose(fit.results["err_f_diode"].value, 376.8348950010331, rtol=1e-4)
    np.testing.assert_allclose(fit.results["err_alpha"].value, 0.014665787938238972, rtol=1e-4)
    np.testing.assert_allclose(fit.results["chi_squared_per_deg"].value, 0.8692224693465523)
    np.testing.assert_allclose(fit.results["backing"].value, 14.919053123539882)


def test_distance_to_surface_input(integration_test_parameters):
    signal = np.cos(2 * np.pi * 37 * np.arange(0, 1, 1.0 / 78125))
    pars = {"bead_diameter": 1.0,
            "driving_data": signal,
            "force_voltage_data": signal,
            "driving_frequency_guess": 37,
            "sample_rate": 78125}

    with pytest.raises(ValueError):
        ActiveCalibrationModel(distance_to_surface=0.49, hydrodynamically_correct=True, **pars)

    # Passes because it's unused when not using hydro model
    ActiveCalibrationModel(distance_to_surface=0.49, hydrodynamically_correct=False, **pars)

    # Distance passes the check
    ActiveCalibrationModel(distance_to_surface=0.51, hydrodynamically_correct=True, **pars)

    # Passes because we're not using the distance in this case.
    ActiveCalibrationModel(distance_to_surface=None, hydrodynamically_correct=True, **pars)
