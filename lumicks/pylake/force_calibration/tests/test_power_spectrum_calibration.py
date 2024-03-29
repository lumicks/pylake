import os
import re
from textwrap import dedent

import numpy as np
import scipy as sp
import pytest

from lumicks.pylake.force_calibration import power_spectrum_calibration as psc
from lumicks.pylake.force_calibration.calibration_models import (
    NoFilter,
    PassiveCalibrationModel,
    density_of_water,
    viscosity_of_water,
    sphere_friction_coefficient,
)
from lumicks.pylake.force_calibration.detail.salty_water import (
    molality_to_molarity,
    molarity_to_molality,
    _density_of_salt_solution,
)

from .data.simulate_calibration_data import generate_active_calibration_test_data


def test_model_parameters():
    params = PassiveCalibrationModel(10, temperature=30)
    assert params.bead_diameter == 10
    assert params.viscosity == viscosity_of_water(30)
    assert params.temperature == 30

    with pytest.raises(TypeError):
        PassiveCalibrationModel(10, invalid_parameter=5)


def test_input_validation_power_spectrum_calibration():
    model = PassiveCalibrationModel(1)

    # Wrong dimensions
    with pytest.raises(TypeError):
        psc.fit_power_spectrum(
            data=np.array([[1, 2, 3], [1, 2, 3]]), sample_rate=78125, model=model
        )

    # Wrong type
    with pytest.raises(TypeError):
        psc.fit_power_spectrum(data="bloop", sample_rate=78125, model=model)

    with pytest.raises(TypeError):
        psc.fit_power_spectrum(data=np.array([1, 2, 3]), sample_rate=78125, model="invalid")

    with pytest.raises(TypeError):
        psc.fit_power_spectrum(
            data=np.array([1, 2, 3]), sample_rate=78125, model=model, settings="invalid"
        )


def test_calibration_result():
    with pytest.raises(TypeError):
        psc.CalibrationResults(invalid=5)


@pytest.mark.parametrize(
    "loss_function, corner_frequency,diffusion_constant,alpha,f_diode,num_samples,viscosity,bead_diameter,temperature",
    [
        ["gaussian", 1000, 1e-9, 0.5, 10000, 30000, 1.002e-3, 4.0, 20.0],
        ["gaussian", 1500, 1.2e-9, 0.5, 10000, 50000, 1.002e-3, 4.0, 20.0],
        ["gaussian", 1500, 1.2e-9, 0.5, 5000, 30000, 1.002e-3, 4.0, 20.0],
        ["gaussian", 1500, 1.2e-9, 0.5, 5000, 30000, 1.2e-3, 4.0, 20.0],
        ["gaussian", 1500, 1.2e-9, 0.5, 5000, 30000, 1.002e-3, 8.0, 20.0],
        ["gaussian", 1500, 1.2e-9, 0.5, 5000, 30000, 1.002e-3, 4.0, 34.0],
        ["gaussian", 1000, 1e-9, 0.5, 10000, 30000, 1, 4.0, 20.0],
        ["lorentzian", 1000, 1e-9, 0.5, 10000, 30000, 1.002e-3, 4.0, 20.0],
        ["lorentzian", 1500, 1.2e-9, 0.5, 10000, 50000, 1.002e-3, 4.0, 20.0],
        ["lorentzian", 1500, 1.2e-9, 0.5, 5000, 30000, 1.002e-3, 4.0, 20.0],
        ["lorentzian", 1500, 1.2e-9, 0.5, 5000, 30000, 1.2e-3, 4.0, 20.0],
        ["lorentzian", 1500, 1.2e-9, 0.5, 5000, 30000, 1.002e-3, 8.0, 20.0],
        ["lorentzian", 1500, 1.2e-9, 0.5, 5000, 30000, 1.002e-3, 4.0, 34.0],
        ["lorentzian", 1000, 1e-9, 0.5, 10000, 30000, 1, 4.0, 20.0],
    ],
)
def test_good_fit_integration(
    compare_to_reference_dict,
    reference_models,
    loss_function,
    corner_frequency,
    diffusion_constant,
    alpha,
    f_diode,
    num_samples,
    viscosity,
    bead_diameter,
    temperature,
):
    data, f_sample = reference_models.lorentzian_td(
        corner_frequency, diffusion_constant, alpha, f_diode, num_samples
    )
    model = PassiveCalibrationModel(bead_diameter, temperature=temperature, viscosity=viscosity)
    power_spectrum = psc.calculate_power_spectrum(
        data, f_sample, fit_range=(0, 15000), num_points_per_block=20
    )
    ps_calibration = psc.fit_power_spectrum(
        power_spectrum=power_spectrum,
        model=model,
        bias_correction=False,
        loss_function=loss_function,
    )

    np.testing.assert_allclose(ps_calibration["fc"].value, corner_frequency, rtol=1e-4)
    np.testing.assert_allclose(ps_calibration["D"].value, diffusion_constant, rtol=1e-4, atol=0)
    np.testing.assert_allclose(ps_calibration["alpha"].value, alpha, rtol=1e-4)
    np.testing.assert_allclose(ps_calibration["f_diode"].value, f_diode, rtol=1e-4)

    gamma = sphere_friction_coefficient(viscosity, bead_diameter * 1e-6)
    kappa_true = 2.0 * np.pi * gamma * corner_frequency * 1e3
    boltzmann_temperature = sp.constants.k * sp.constants.convert_temperature(temperature, "C", "K")
    rd_true = np.sqrt(boltzmann_temperature / gamma / diffusion_constant) * 1e6
    np.testing.assert_allclose(ps_calibration["kappa"].value, kappa_true, rtol=1e-4)
    np.testing.assert_allclose(ps_calibration["Rd"].value, rd_true, rtol=1e-4)
    np.testing.assert_allclose(ps_calibration["Rf"].value, rd_true * kappa_true * 1e3, rtol=1e-4)
    np.testing.assert_allclose(
        ps_calibration["chi_squared_per_deg"].value, 0, atol=1e-9
    )  # Noise free

    if loss_function == "gaussian":
        compare_to_reference_dict(
            {
                par: ps_calibration[par].value
                for par in ("err_fc", "err_D", "err_f_diode", "err_alpha", "err_kappa", "err_Rd")
            }
        )


def test_fit_settings(reference_models):
    """This test tests whether the algorithm parameters ftol, max_function_evals and
    analytical_fit_range for lk.fit_power_spectrum() are applied as intended."""
    sample_rate = 78125
    corner_frequency, diffusion_volt = 4000, 1.14632
    bead_diameter, temperature, viscosity = 1.03, 20, 1.002e-3

    # alpha = 1.0 means no diode effect
    data, f_sample = reference_models.lorentzian_td(
        corner_frequency, diffusion_volt, alpha=1.0, f_diode=14000, num_samples=sample_rate
    )
    model = PassiveCalibrationModel(bead_diameter, temperature=temperature, viscosity=viscosity)
    power_spectrum = psc.calculate_power_spectrum(
        data, f_sample, fit_range=(0, 23000), num_points_per_block=200
    )

    # Won't converge with so few maximum function evaluations
    with pytest.raises(
        RuntimeError, match="The maximum number of function evaluations is exceeded"
    ):
        psc.fit_power_spectrum(power_spectrum=power_spectrum, model=model, max_function_evals=1)

    # Make the analytical fit fail
    with pytest.raises(
        RuntimeError, match="An empty power spectrum was passed to fit_analytical_lorentzian"
    ):
        psc.fit_power_spectrum(
            power_spectrum=power_spectrum, model=model, analytical_fit_range=(10, 100)
        )


def test_bad_calibration_result_arg():
    with pytest.raises(TypeError):
        psc.CalibrationResults(bad_arg=5)


def test_no_data_in_range():
    model = PassiveCalibrationModel(1, temperature=20, viscosity=0.0004)

    # Here the range slices off all the data and we are left with an empty spectrum
    power_spectrum = psc.PowerSpectrum(np.arange(100), sample_rate=100).in_range(47, 100)

    with pytest.raises(RuntimeError):
        psc.fit_power_spectrum(power_spectrum, model=model)

    # Check whether a failure to get a sufficient number of points in the analytical fit is
    # detected.
    power_spectrum = psc.PowerSpectrum(np.arange(100), sample_rate=1e-3)

    with pytest.raises(RuntimeError):
        psc.fit_power_spectrum(power_spectrum, model=model)


@pytest.fixture(scope="module")
def reference_calibration_result():
    data = np.load(os.path.join(os.path.dirname(__file__), "data/reference_spectrum.npz"))
    reference_spectrum = data["arr_0"]
    model = PassiveCalibrationModel(4.4, temperature=20, viscosity=0.001002)
    reference_spectrum = psc.calculate_power_spectrum(
        reference_spectrum, sample_rate=78125, num_points_per_block=100, fit_range=(100.0, 23000.0)
    )
    ps_calibration = psc.fit_power_spectrum(
        power_spectrum=reference_spectrum, model=model, bias_correction=False
    )

    return ps_calibration, model, reference_spectrum


def test_bad_fit(reference_calibration_result):
    ps_calibration, model, reference_spectrum = reference_calibration_result
    bad_spectrum = reference_spectrum.power.copy()
    bad_spectrum[30:31] = reference_spectrum.power[10]  # Chop!
    bad_spectrum = reference_spectrum.with_spectrum(
        bad_spectrum, num_points_per_block=reference_spectrum.num_points_per_block
    )
    bad_calibration = psc.fit_power_spectrum(
        power_spectrum=bad_spectrum, model=model, loss_function="gaussian"
    )

    assert ps_calibration["backing"].value > bad_calibration["backing"].value


def test_actual_spectrum(reference_calibration_result):
    ps_calibration, model, reference_spectrum = reference_calibration_result

    results = {
        "D": {"desired": 0.0018512505734895896, "rtol": 1e-4, "atol": 0},
        "Rd": {"desired": 7.253677199344564, "rtol": 1e-4},
        "Rf": {"desired": 1243.966729922322, "rtol": 1e-4},
        "kappa": {"desired": 0.17149463585651784, "rtol": 1e-4},
        "alpha": {"desired": 0.5006070381347969, "rtol": 1e-4},
        "backing": {"desired": 30.570451, "rtol": 1e-4},
        "chi_squared_per_deg": {"desired": 1.0637833024139873, "rtol": 1e-4},
        "err_fc": {"desired": 32.22822335114943, "rtol": 1e-4},
        "err_D": {"desired": 6.429704886151389e-05, "rtol": 1e-4, "atol": 0},
        "err_alpha": {"desired": 0.013140824804884007, "rtol": 1e-4},
        "err_f_diode": {"desired": 561.7212147994059, "rtol": 1e-4},
    }

    for name, expected_result in results.items():
        np.testing.assert_allclose(ps_calibration[name].value, **expected_result)
        np.testing.assert_allclose(ps_calibration.results[name].value, **expected_result)

    params = {
        "Viscosity": {"desired": 0.001002},
        "Temperature": {"desired": 20},
        "Max iterations": {"desired": 10000},
        "Fit tolerance": {"desired": 1e-07},
        "Points per block": {"desired": 100},
        "Sample rate": {"desired": 78125},
    }

    for name, expected_result in params.items():
        np.testing.assert_allclose(ps_calibration[name].value, **expected_result)
        np.testing.assert_allclose(ps_calibration.params[name].value, **expected_result)

    # Test whether the model contains the number of points per block that were used to fit it
    np.testing.assert_allclose(ps_calibration.ps_model.num_points_per_block, 100)
    np.testing.assert_allclose(ps_calibration.ps_data.num_points_per_block, 100)


def test_result_plot(reference_calibration_result):
    ps_calibration, model, reference_spectrum = reference_calibration_result
    ps_calibration.plot()


def test_result_plot(reference_calibration_result):
    ps_calibration, model, reference_spectrum = reference_calibration_result
    ps_calibration.plot_spectrum_residual()


def test_attributes_ps_calibration(reference_calibration_result):
    ps_calibration, model, reference_spectrum = reference_calibration_result
    assert id(ps_calibration.model) == id(model)
    assert id(ps_calibration.ps_data) == id(reference_spectrum)

    with pytest.raises(RuntimeError):
        psc.CalibrationResults(
            model=None,
            ps_model=None,
            ps_data=None,
            params={"test": 5},
            results={"test2": 5},
            fitted_params=[],
        )


def test_calibration_results_params():
    result = psc.CalibrationResults(
        model=None,
        ps_model=None,
        ps_data=None,
        params={"test": psc.CalibrationParameter("par", "val", 5)},
        results={
            "Rf": psc.CalibrationParameter("Rf", "val", 5),
            "kappa": psc.CalibrationParameter("kappa", "val", 5),
        },
        fitted_params=[],
    )
    assert "test" in result
    assert "Rf" in result
    assert "nope" not in result


def test_repr(reference_calibration_result):
    ps_calibration, model, reference_spectrum = reference_calibration_result
    assert str(ps_calibration) == dedent(
        """\
        Name                 Description                                               Value
        -------------------  --------------------------------------------------------  -----------
        Bead diameter        Bead diameter (um)                                        4.4
        Viscosity            Liquid viscosity (Pa*s)                                   0.001002
        Temperature          Liquid temperature (C)                                    20
        Distance to surface  Distance from bead center to surface (um)
        Max iterations       Maximum number of function evaluations                    10000
        Fit tolerance        Fitting tolerance                                         1e-07
        Points per block     Number of points per block                                100
        Sample rate          Sample rate (Hz)                                          78125
        Bias correction      Perform bias correction thermal fit                       0
        Loss function        Loss function used during minimization                    gaussian
        Rd                   Distance response (um/V)                                  7.25366
        kappa                Trap stiffness (pN/nm)                                    0.171495
        Rf                   Force response (pN/V)                                     1243.97
        gamma_0              Theoretical bulk drag coefficient (kg/s)                  4.1552e-08
        err_kappa            Stiffness Std Err (pN/V)                                  0.00841414
        err_Rd               Distance response Std Err (um/V)                          0.125966
        fc                   Corner frequency (Hz)                                     656.872
        D                    Diffusion constant (V^2/s)                                0.00185126
        err_fc               Corner frequency Std Err (Hz)                             32.2284
        err_D                Diffusion constant Std Err (V^2/s)                        6.42974e-05
        f_diode              Diode low-pass filtering roll-off frequency (Hz)          7936.51
        alpha                Diode 'relaxation factor'                                 0.500609
        err_f_diode          Diode low-pass filtering roll-off frequency Std Err (Hz)  561.715
        err_alpha            Diode 'relaxation factor' Std Err                         0.0131406
        chi_squared_per_deg  Chi squared per degree of freedom                         1.06378
        backing              Statistical backing (%)                                   30.5705"""
    )


def test_invalid_bead_diameter():
    with pytest.raises(ValueError, match="Invalid bead diameter specified"):
        PassiveCalibrationModel(bead_diameter=0)

    with pytest.raises(ValueError, match="Invalid bead diameter specified"):
        PassiveCalibrationModel(bead_diameter=1e-7)

    PassiveCalibrationModel(bead_diameter=1e-2)


def test_faxen_correction():
    """When hydro is off, but a height is given, the interpretation of the Lorentzian fit can still
    benefit from using the distance to the surface in a correction factor for the drag. This will
    only affect thermal calibration. This behaviour is tested here."""
    shared_pars = {
        "bead_diameter": 1.03,
        "viscosity": 1.1e-3,
        "temperature": 25,
        "rho_sample": 997.0,
        "rho_bead": 1040.0,
        "distance_to_surface": 1.03 / 2 + 400e-3,
    }
    sim_pars = {
        "sample_rate": 78125,
        "stiffness": 0.1,
        "pos_response_um_volt": 0.618,
        "driving_sinusoid": (500, 31.95633),
        "diode": (0.4, 15000),
    }

    np.random.seed(10071985)
    volts, _ = generate_active_calibration_test_data(
        10, hydrodynamically_correct=True, **sim_pars, **shared_pars
    )
    power_spectrum = psc.calculate_power_spectrum(volts, sim_pars["sample_rate"])

    model = PassiveCalibrationModel(**shared_pars, hydrodynamically_correct=False)
    fit = psc.fit_power_spectrum(power_spectrum, model, bias_correction=False)

    # Fitting with *no* hydrodynamically correct model, but *with* Faxen's law
    np.testing.assert_allclose(fit.results["Rd"].value, 0.6136895577998873)
    np.testing.assert_allclose(fit.results["kappa"].value, 0.10312266251783221)
    np.testing.assert_allclose(fit.results["Rf"].value, 63.285301159715466)
    np.testing.assert_allclose(fit.results["gamma_0"].value, 1.0678273429551705e-08)

    # Disabling Faxen's correction on the drag makes the estimates *much* worse
    shared_pars["distance_to_surface"] = None
    model = PassiveCalibrationModel(**shared_pars, hydrodynamically_correct=False)
    fit = psc.fit_power_spectrum(power_spectrum, model, bias_correction=False)
    np.testing.assert_allclose(fit.results["Rd"].value, 0.741747603986908)
    np.testing.assert_allclose(fit.results["kappa"].value, 0.07058936587810064)
    np.testing.assert_allclose(fit.results["Rf"].value, 52.35949300703634)
    # Not affected since this is gamma bulk
    np.testing.assert_allclose(fit.results["gamma_0"].value, 1.0678273429551705e-08)


@pytest.mark.parametrize("temperatures", [np.arange(-30, -10), np.arange(100, 130), -21, 110])
def test_viscosity_calculation_invalid_range(temperatures):
    with pytest.raises(
        ValueError, match="Function for viscosity of water is only valid for -20°C <= T < 110°C"
    ):
        viscosity_of_water(temperatures)


def test_viscosity_calculation():
    temperatures = np.arange(20, 100, 10)

    ref = np.array(
        [
            0.0010015672646030015,
            0.0007972062022678064,
            0.0006527334742093661,
            0.0005465265005450516,
            0.00046603929414699226,
            0.000403545426497709,
            0.000354046106187348,
            0.0003141732579601075,
        ]
    )

    np.testing.assert_allclose(viscosity_of_water(temperatures), ref)
    np.testing.assert_allclose(viscosity_of_water(list(temperatures)), ref)
    assert viscosity_of_water(temperatures).shape == ref.shape

    np.testing.assert_allclose(viscosity_of_water(20), 0.0010015672646030015)
    assert viscosity_of_water(20).shape == ()


@pytest.mark.parametrize(
    "temperature, pressure",
    (
        (20, 0.101325),
        (25, 0.101325),
        (45, 0.401325),
        (40, 0.201325),
        (130, 15),
        (130, 5),
    ),
)
def test_molality_molarity_conversions(temperature, pressure):
    molarities = np.hstack((np.arange(0, 5, 0.1), np.arange(0, 0.2, 0.01)))
    round_trip = [
        molality_to_molarity(
            molarity_to_molality(c, temperature, pressure, molecular_weight=58.4428),
            temperature,
            pressure,
            molecular_weight=58.4428,
        )
        for c in molarities
    ]
    np.testing.assert_allclose(molarities, round_trip)


def test_molarity_molality_conversion_out_of_range():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "outside the valid range of the solution density model (0.0, 5.310637057482627)"
        ),
    ):
        molarity_to_molality(5.32, 23, 0.1, molecular_weight=58.4428)

    molarity_to_molality(5.31, 23, 0.1, molecular_weight=58.4428)


@pytest.mark.parametrize(
    "condition, temperature, molality, pressure, ref_value, ref_kinematic",
    [
        ("reference", 20, 0, 0.1, 1002.0, 1.0037),
        ("temp", 40, 0, 0.1, 652.9, 0.6580),
        ("pressure", 20, 0, 15.0, 996.1, 0.9911),
        ("pressure_temp", 40, 0, 15.0, 654.3, 0.6551),
        ("salt", 20, 0.5, 0.1, 1043.2, 1.0244),
        ("salt_temp", 40, 0.5, 0.1, 684.8, 0.6769),
        ("salt_pressure", 20, 0.5, 10.0, 1041.1, 1.0179),
        ("salt_pressure_temp", 40, 0.5, 10.0, 686.5, 0.6757),
        ("salty_pressure_temp", 40, 1.0, 10.0, 722.0, 0.6978),
        ("extreme_conditions", 20, 5.0, 0.1, 1714.8, 1.4661),
        ("extreme_conditions2", 20, 5.0, 15, 1729.6, 1.4721),
        ("extreme_conditions3", 100, 5.0, 15, 508.2, 0.4506),
        ("extreme_conditions3", 100, 5.0, 20, 510.3, 0.4517),
        ("middle_everything", 80, 3.0, 10, 503.0, 0.4658),
        ("cook", 130, 3.0, 15, 310.3, 0.2958),
    ],
)
def test_viscosity_salty_water(
    condition, temperature, molality, pressure, ref_value, ref_kinematic
):
    molarity = molality_to_molarity(molality, temperature, pressure, molecular_weight=58.4428)
    viscosity = viscosity_of_water(temperature, molarity, pressure)
    np.testing.assert_allclose(
        viscosity,
        ref_value / 1e6,
        rtol=5e-4,  # Reference values in the paper were reported with this tolerance
        err_msg=condition,
    )
    np.testing.assert_allclose(
        viscosity / density_of_water(temperature, molarity, pressure),
        ref_kinematic / 1e6,
        rtol=1e-3,  # Reference values in the paper were reported with this tolerance
        err_msg=condition,
    )
    np.testing.assert_allclose(
        viscosity / _density_of_salt_solution(temperature, molality, pressure),
        ref_kinematic / 1e6,
        rtol=1e-3,  # Reference values in the paper were reported with this tolerance
        err_msg=condition,
    )


def test_viscosity_temperature_array():
    temperature = np.arange(20, 40, 5)
    ref_viscosity = [
        0.00100977454389667,
        0.0008974568504064918,
        0.0008041499558427885,
        0.0007257335507374668,
    ]
    np.testing.assert_allclose(viscosity_of_water(temperature, 0.1, 0.1), ref_viscosity)


def test_density_temperature_array():
    temperature = np.arange(20, 40, 5)
    ref_density = [
        1002.325455422484,
        1001.1379828996089,
        999.7116776292133,
        998.0679282931027,
    ]
    np.testing.assert_allclose(density_of_water(temperature, 0.1), ref_density)


def test_invalid_densities():
    """Densities lower than aerogel should not be accepted."""
    with pytest.raises(
        ValueError, match=re.escape("Density of the sample cannot be below 100 kg/m^3")
    ):
        PassiveCalibrationModel(4.1, hydrodynamically_correct=True, rho_sample=99.9)

    # Make sure 0 also fires (since it's falsy)
    with pytest.raises(
        ValueError, match=re.escape("Density of the sample cannot be below 100 kg/m^3")
    ):
        PassiveCalibrationModel(4.1, hydrodynamically_correct=True, rho_sample=0)

    with pytest.raises(
        ValueError, match=re.escape("Density of the bead cannot be below 100 kg/m^3")
    ):
        PassiveCalibrationModel(4.1, hydrodynamically_correct=True, rho_bead=99.9)

    PassiveCalibrationModel(4.1, hydrodynamically_correct=True, rho_sample=100.1)
    PassiveCalibrationModel(4.1, hydrodynamically_correct=True, rho_bead=100.1)

    # When not using hydro, these arguments are not used (no need to raise).
    PassiveCalibrationModel(4.1, hydrodynamically_correct=False, rho_sample=99.9)
    PassiveCalibrationModel(4.1, hydrodynamically_correct=False, rho_bead=99.9)


def test_invalid_viscosity():
    with pytest.raises(ValueError, match=re.escape("Viscosity must be higher than 0.0003 Pa*s")):
        PassiveCalibrationModel(4.1, viscosity=0.0003)

    with pytest.raises(ValueError, match=re.escape("Viscosity must be higher than 0.0003 Pa*s")):
        PassiveCalibrationModel(4.1, viscosity=0)

    PassiveCalibrationModel(4.1, viscosity=0.00031)


def test_invalid_temperature():
    with pytest.raises(ValueError, match=re.escape("Temperature must be between 5 and 90 Celsius")):
        PassiveCalibrationModel(4.1, temperature=90.0)

    with pytest.raises(ValueError, match=re.escape("Temperature must be between 5 and 90 Celsius")):
        PassiveCalibrationModel(4.1, temperature=5.0)

    PassiveCalibrationModel(4.1, temperature=89.9)
    PassiveCalibrationModel(4.1, temperature=5.1)


def test_invalid_distance_to_surface():
    # Check that zero does not work specifically (since it is falsy).
    with pytest.raises(
        ValueError,
        match="Distance from bead center to surface is smaller than the bead radius",
    ):
        PassiveCalibrationModel(4.11, distance_to_surface=0, hydrodynamically_correct=False)


@pytest.mark.slow
def test_aliasing(integration_test_parameters):
    """Test whether the private API for taking into account aliasing works"""
    shared_pars, simulation_pars = integration_test_parameters

    np.random.seed(10071985)
    decimation_factor = 10
    shared_pars["hydrodynamically_correct"] = False
    simulation_pars["sample_rate"] = decimation_factor * simulation_pars["sample_rate"]
    volts, _ = generate_active_calibration_test_data(5, **simulation_pars, **shared_pars)

    # Drop the sample rate by a factor without taking it into account -> results in aliasing
    aliased_samplerate = simulation_pars["sample_rate"] // decimation_factor
    aliased_volts = volts[::decimation_factor]
    power_spectrum = psc.calculate_power_spectrum(
        aliased_volts, aliased_samplerate, fit_range=(100, aliased_samplerate / 2)
    )

    model = PassiveCalibrationModel(**shared_pars)
    sim_args = [np.arange(22050), 500, 1e-3, 1400, 0.4]
    ref_sim = model(*sim_args)
    aliased_model = model._alias_model(sample_rate=aliased_samplerate, num_aliases=10)

    # Make sure basic model did not change
    np.testing.assert_allclose(ref_sim, model(*sim_args))

    fit = psc.fit_power_spectrum(power_spectrum, model, bias_correction=False)

    # Check whether modification of the base model does not affect the aliased model
    model._filter = NoFilter()
    model._set_drag(5)

    correct_fit = psc.fit_power_spectrum(power_spectrum, aliased_model, bias_correction=False)

    expected_results = {"fc": 1504.4416105821158, "D": 1.0090151317063}

    for key, value in expected_results.items():
        assert abs(fit.results[key].value - value) > abs(correct_fit.results[key].value - value)
