from lumicks.pylake.force_calibration import power_spectrum_calibration as psc
from lumicks.pylake.force_calibration.calibration_models import PassiveCalibrationModel, sphere_friction_coefficient
from matplotlib.testing.decorators import cleanup
from textwrap import dedent
from copy import deepcopy
import numpy as np
import scipy as sp
import os
import pytest


def test_model_parameters():
    params = PassiveCalibrationModel(10, temperature=30)
    assert params.bead_diameter == 10
    assert params.viscosity == 1.002e-3
    assert params.temperature == 30

    with pytest.raises(TypeError):
        PassiveCalibrationModel(10, invalid_parameter=5)


def test_input_validation_power_spectrum_calibration():
    model = PassiveCalibrationModel(1)

    # Wrong dimensions
    with pytest.raises(TypeError):
        psc.fit_power_spectrum(data=np.array([[1, 2, 3], [1, 2, 3]]), sample_rate=78125, model=model)

    # Wrong type
    with pytest.raises(TypeError):
        psc.fit_power_spectrum(data="bloop", sample_rate=78125, model=model)

    with pytest.raises(TypeError):
        psc.fit_power_spectrum(data=np.array([1, 2, 3]), sample_rate=78125, model="invalid")

    with pytest.raises(TypeError):
        psc.fit_power_spectrum(data=np.array([1, 2, 3]), sample_rate=78125, model=model, settings="invalid")


def test_calibration_result():
    with pytest.raises(TypeError):
        psc.CalibrationResults(invalid=5)


@pytest.mark.parametrize(
    "corner_frequency,diffusion_constant,alpha,f_diode,num_samples,viscosity,bead_diameter,temperature,err_fc,err_d,"
    "err_f_diode,err_alpha,",
    [
        [1000, 1e-9, 0.5, 10000, 30000, 1.002e-3, 4.0, 20.0, 29.77266, 2.984664e-11, 1239.061833, 0.05615039],
        [1500, 1.2e-9, 0.5, 10000, 50000, 1.002e-3, 4.0, 20.0, 47.2181, 4.589085e-11, 1399.049903, 0.05856517],
        [1500, 1.2e-9, 0.5, 5000, 30000, 1.002e-3, 4.0, 20.0, 70.59478, 8.226641e-11, 487.4102, 0.01342818],
        [1500, 1.2e-9, 0.5, 5000, 30000, 1.2e-3, 4.0, 20.0, 70.59478, 8.226641e-11, 487.4102, 0.01342818],
        [1500, 1.2e-9, 0.5, 5000, 30000, 1.002e-3, 8.0, 20.0, 70.59478, 8.226641e-11, 487.4102, 0.01342818],
        [1500, 1.2e-9, 0.5, 5000, 30000, 1.002e-3, 4.0, 34.0, 70.59478, 8.226641e-11, 487.4102, 0.01342818],
        [1000, 1e-9, 0.5, 10000, 30000, 1.002e-3, 4.0, 20.0, 29.77266, 2.984664e-11, 1239.061833, 0.05615039],
        [1000, 1e-9, 0.5, 10000, 30000, 1, 4.0, 20.0, 29.77266, 2.984664e-11, 1239.061833, 0.05615039],
    ],
)
def test_good_fit_integration_test(
    reference_models,
    corner_frequency,
    diffusion_constant,
    alpha,
    f_diode,
    num_samples,
    viscosity,
    bead_diameter,
    temperature,
    err_fc,
    err_d,
    err_f_diode,
    err_alpha,
):
    data, f_sample = reference_models.lorentzian_td(corner_frequency, diffusion_constant, alpha, f_diode, num_samples)
    model = PassiveCalibrationModel(bead_diameter, temperature=temperature, viscosity=viscosity)
    power_spectrum = psc.calculate_power_spectrum(data, f_sample, fit_range=(0, 15000), num_points_per_block=20)
    ps_calibration = psc.fit_power_spectrum(power_spectrum=power_spectrum, model=model, bias_correction=False)

    np.testing.assert_allclose(ps_calibration["fc"].value, corner_frequency, rtol=1e-4)
    np.testing.assert_allclose(ps_calibration["D"].value, diffusion_constant, rtol=1e-4, atol=0)
    np.testing.assert_allclose(ps_calibration["alpha"].value, alpha, rtol=1e-4)
    np.testing.assert_allclose(ps_calibration["f_diode"].value, f_diode, rtol=1e-4)

    gamma = sphere_friction_coefficient(viscosity, bead_diameter * 1e-6)
    kappa_true = 2.0 * np.pi * gamma * corner_frequency * 1e3
    rd_true = (
        np.sqrt(sp.constants.k * sp.constants.convert_temperature(temperature, "C", "K") / gamma / diffusion_constant)
        * 1e6
    )
    np.testing.assert_allclose(ps_calibration["kappa"].value, kappa_true, rtol=1e-4)
    np.testing.assert_allclose(ps_calibration["Rd"].value, rd_true, rtol=1e-4)
    np.testing.assert_allclose(ps_calibration["Rf"].value, rd_true * kappa_true * 1e3, rtol=1e-4)
    np.testing.assert_allclose(ps_calibration["chi_squared_per_deg"].value, 0, atol=1e-9)  # Noise free

    np.testing.assert_allclose(ps_calibration["err_fc"].value, err_fc)
    np.testing.assert_allclose(ps_calibration["err_D"].value, err_d, rtol=1e-4, atol=0)
    np.testing.assert_allclose(ps_calibration["err_f_diode"].value, err_f_diode)
    np.testing.assert_allclose(ps_calibration["err_alpha"].value, err_alpha, rtol=1e-6)


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
    with pytest.raises(RuntimeError, match="The maximum number of function evaluations is exceeded"):
        psc.fit_power_spectrum(power_spectrum=power_spectrum, model=model, max_function_evals=1)

    # Make the analytical fit fail
    with pytest.raises(RuntimeError, match="An empty power spectrum was passed to fit_analytical_lorentzian"):
        psc.fit_power_spectrum(power_spectrum=power_spectrum, model=model, analytical_fit_range=(10, 100))


def test_bad_calibration_result_arg():
    with pytest.raises(TypeError):
        psc.CalibrationResults(bad_arg=5)


def test_bad_data():
    num_samples = 30000
    data = np.sin(.1*np.arange(num_samples))
    model = PassiveCalibrationModel(1, temperature=20, viscosity=0.0001)
    power_spectrum = psc.PowerSpectrum(data, num_samples)

    with pytest.raises(ValueError):
        psc.fit_power_spectrum(power_spectrum, model=model)


def test_no_data_in_range():
    model = PassiveCalibrationModel(1, temperature=20, viscosity=0.0001)

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
    data = np.load(os.path.join(os.path.dirname(__file__), "reference_spectrum.npz"))
    reference_spectrum = data["arr_0"]
    model = PassiveCalibrationModel(4.4, temperature=20, viscosity=0.001002)
    reference_spectrum = psc.calculate_power_spectrum(reference_spectrum, sample_rate=78125,
                                                      num_points_per_block=100,
                                                      fit_range=(100.0, 23000.0))
    ps_calibration = psc.fit_power_spectrum(power_spectrum=reference_spectrum,
                                            model=model,
                                            bias_correction=False)

    return ps_calibration, model, reference_spectrum


def test_actual_spectrum(reference_calibration_result):
    ps_calibration, model, reference_spectrum = reference_calibration_result

    results = {
        "D": {"desired": 0.0018512505734895896, "rtol": 1e-4, "atol": 0},
        "Rd": {"desired": 7.253677199344564, "rtol": 1e-4},
        "Rf": {"desired": 1243.966729922322, "rtol": 1e-4},
        "kappa": {"desired": 0.17149463585651784, "rtol": 1e-4},
        "alpha": {"desired": 0.5006070381347969, "rtol": 1e-4},
        "backing": {"desired": 66.43310564863437, "rtol": 1e-4},
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
        "Sample rate": {"desired": 78125}
    }

    for name, expected_result in params.items():
        np.testing.assert_allclose(ps_calibration[name].value, **expected_result)
        np.testing.assert_allclose(ps_calibration.params[name].value, **expected_result)

    # Test whether the model contains the number of points per block that were used to fit it
    np.testing.assert_allclose(ps_calibration.ps_model.num_points_per_block, 100)
    np.testing.assert_allclose(ps_calibration.ps_data.num_points_per_block, 100)


@cleanup
def test_result_plot(reference_calibration_result):
    ps_calibration, model, reference_spectrum = reference_calibration_result
    ps_calibration.plot()


@cleanup
def test_result_plot(reference_calibration_result):
    ps_calibration, model, reference_spectrum = reference_calibration_result
    ps_calibration.plot_spectrum_residual()


def test_attributes_ps_calibration(reference_calibration_result):
    ps_calibration, model, reference_spectrum = reference_calibration_result
    assert id(ps_calibration.model) == id(model)
    assert id(ps_calibration.ps_data) == id(reference_spectrum)

    with pytest.raises(RuntimeError):
        psc.CalibrationResults(model=None,
                               ps_model=None,
                               ps_data=None,
                               params={"test": 5},
                               results={"test2": 5})


def test_repr(reference_calibration_result):
    ps_calibration, model, reference_spectrum = reference_calibration_result
    assert str(ps_calibration) == dedent("""\
        Name                 Description                                                         Value
        -------------------  --------------------------------------------------------  ---------------
        Bead diameter        Bead diameter (um)                                            4.4
        Viscosity            Liquid viscosity (Pa*s)                                       0.001002
        Temperature          Liquid temperature (C)                                       20
        Max iterations       Maximum number of function evaluations                    10000
        Fit tolerance        Fitting tolerance                                             1e-07
        Points per block     Number of points per block                                  100
        Sample rate          Sample rate (Hz)                                          78125
        Bias correction      Perform bias correction thermal fit                           0
        Rd                   Distance response (um/V)                                      7.25366
        kappa                Trap stiffness (pN/nm)                                        0.171495
        Rf                   Force response (pN/V)                                      1243.97
        gamma_0              Theoretical drag coefficient (kg/s)                           4.1552e-08
        fc                   Corner frequency (Hz)                                       656.872
        D                    Diffusion constant (V^2/s)                                    0.00185126
        err_fc               Corner frequency Std Err (Hz)                                32.2284
        err_D                Diffusion constant Std Err (V^2/s)                            6.42974e-05
        f_diode              Diode low-pass filtering roll-off frequency (Hz)           7936.51
        alpha                Diode 'relaxation factor'                                     0.500609
        err_f_diode          Diode low-pass filtering roll-off frequency Std Err (Hz)    561.715
        err_alpha            Diode 'relaxation factor' Std Err                             0.0131406
        chi_squared_per_deg  Chi squared per degree of freedom                             1.06378
        backing              Statistical backing (%)                                      66.4331""")


def test_invalid_bead_diameter():
    with pytest.raises(ValueError, match="Invalid bead diameter specified"):
        PassiveCalibrationModel(bead_diameter=0)

    with pytest.raises(ValueError, match="Invalid bead diameter specified"):
        PassiveCalibrationModel(bead_diameter=1e-7)

    PassiveCalibrationModel(bead_diameter=1e-2)
