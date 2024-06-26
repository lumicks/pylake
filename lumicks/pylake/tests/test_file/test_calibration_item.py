import pytest

from lumicks.pylake.calibration import ForceCalibrationItem
from lumicks.pylake.force_calibration.convenience import calibrate_force
from lumicks.pylake.force_calibration.power_spectrum_calibration import calculate_power_spectrum

ref_passive_fixed_diode_with_height = {
    "Axial calibration": 0.0,
    "Bead density (Kg/m3)": 1100.0,  # Deliberately set a non-default
    "Bead diameter (um)": 1.0,
    "D (V^2/s)": 1.5450928303539805e-12,
    "Diode alpha": 0.45,
    "Diode alpha delta": 0.05,
    "Diode alpha max": 0.5,
    "Diode alpha rate": 1.0,
    "Diode frequency (Hz)": 10000.0,
    "Diode frequency delta": 4000.0,
    "Diode frequency max": 14000.0,
    "Diode frequency rate": 1.25,
    "Exclusion range 0 (max.) (Hz)": 500.0,
    "Exclusion range 0 (min.) (Hz)": 100.0,
    "Exclusion range 1 (max.) (Hz)": 2000.0,
    "Exclusion range 1 (min.) (Hz)": 1000.0,
    "Fit range (max.) (Hz)": 23000.0,
    "Fit range (min.) (Hz)": 100.0,
    "Fit tolerance": 1e-07,
    "Fluid density (Kg/m3)": 1000.0,  # Deliberately set a non-default
    "Hydrodynamic correction enabled": 1.0,
    "Kind": "Full calibration",
    "Max iterations": 10000.0,
    "Number of samples": 781250.0,
    "Offset (pN)": 0.0,
    "Points per block": 2000.0,
    "Rd (um/V)": 563574.3158653651,
    "Response (pN/V)": 136444558.55705503,
    "Rf (pN/V)": 136444558.55705503,
    "Sample rate (Hz)": 78125.0,
    "Sign": 1.0,
    "Start time (ns)": 1696171376701856700,
    "Stop time (ns)": 1696171386701856700,
    "Temperature (C)": 25.0,
    "Trap sum power (V)": 0.0,
    "Viscosity (Pa*s)": 0.00089,
    "backing (%)": 100.0,
    "chi_squared_per_deg": 199.87179339908212,
    "err_D (V^2/s)": 4.0980394310000066e-15,
    "err_fc (Hz)": 17.4576138032815,
    "fc (Hz)": 4593.7147688881405,
    "gamma_0 (kg/s)": 8.388052385084746e-09,
    "kappa (pN/nm)": 0.2421057076519628,
    "Bead center height (um)": 1.0,
}


ref_active = {
    "Axial calibration": 0.0,
    "Bead density (Kg/m3)": 1060.0,
    "Bead diameter (um)": 2.1,
    "D (V^2/s)": 0.4966057306587712,
    "Driving data frequency (Hz)": 17.0,
    "Exclusion range 0 (max.) (Hz)": 22.0,
    "Exclusion range 0 (min.) (Hz)": 12.0,
    "Exclusion range 1 (max.) (Hz)": 265.0,
    "Exclusion range 1 (min.) (Hz)": 205.0,
    "Fit range (max.) (Hz)": 23000.0,
    "Fit range (min.) (Hz)": 100.0,
    "Fit tolerance": 1e-07,
    "Fluid density (Kg/m3)": 997.0,
    "Hydrodynamic correction enabled": 1.0,
    "Kind": "Full calibration",
    "Max iterations": 10000.0,
    "Number of samples": 779746.0,
    "Offset (pN)": 0.0,
    "Points per block": 200.0,
    "Rd (um/V)": 0.955652912470082,
    "Response (pN/V)": 481.0567789160435,
    "Rf (pN/V)": 481.0567789160435,
    "Sample rate (Hz)": 78125.0,
    "Sign": -1.0,
    "Start time (ns)": 1713785826919398000,
    "Stop time (ns)": 1713785836900152600,
    "Temperature (C)": 23.0,
    "Viscosity (Pa*s)": 0.00089,
    "alpha": 0.13451116038806527,
    "backing (%)": 73.78628044632126,
    "chi_squared_per_deg": 1.0274348944981855,
    "driving_amplitude (um)": 0.5448874735882269,
    "driving_frequency (Hz)": 16.994852531751125,
    "driving_power (V^2)": 6.042495724703225e-07,
    "err_D (V^2/s)": 0.015338943387826319,
    "err_alpha": 0.6218064777515927,
    "err_f_diode (Hz)": 4758.019964821415,
    "err_fc (Hz)": 150.01565141883069,
    "f_diode (Hz)": 27498.22833357482,
    "fc (Hz)": 8886.553616671345,
    "gamma_0 (kg/s)": 1.761491000867797e-08,
    "gamma_ex (kg/s)": 9.01535675752682e-09,
    "kappa (pN/nm)": 0.5033802258527661,
}


ref_axial = {
    "Axial calibration": 1.0,
    "Bead center height (um)": 0.6,
    "Bead diameter (um)": 1.0,
    "D (V^2/s)": 0.0017731114594436534,
    "Fit range (max.) (Hz)": 7000.0,
    "Fit range (min.) (Hz)": 10.0,
    "Fit tolerance": 1e-07,
    "Hydrodynamic correction enabled": 0.0,
    "Kind": "Full calibration",
    "Max iterations": 10000.0,
    "Number of samples": 780000.0,
    "Offset (pN)": -34.171076858725584,
    "Points per block": 2000.0,
    "Rd (um/V)": 6.496731189269446,
    "Response (pN/V)": 403.89198808558183,
    "Rf (pN/V)": 403.89198808558183,
    "Sample rate (Hz)": 78125.0,
    "Sign": 1.0,
    "Start time (ns)": 1706551902833197700,
    "Stop time (ns)": 1706551912817197700,
    "Temperature (C)": 25.0,
    "Viscosity (Pa*s)": 0.00089,
    "backing (%)": 100.0,
    "chi_squared_per_deg": 163.52988873216955,
    "err_D (V^2/s)": 1.8100005384222422e-05,
    "err_fc (Hz)": 2.9241042944956757,
    "fc (Hz)": 179.8863491705443,
    "gamma_ex_lateral (kg/s)": 8.666798296149962e-09,  # <== specific to axial
    "kappa (pN/nm)": 0.062168493095833215,
}


def test_passive_item(compare_to_reference_dict, calibration_data):
    item = ForceCalibrationItem(ref_passive_fixed_diode_with_height)
    assert item.excluded_ranges == [(100.0, 500.0), (1000.0, 2000.0)]
    assert item.fit_range == (100.0, 23000.0)
    assert item.num_points_per_block == 2000
    assert item.sample_rate == 78125
    assert not item.active_calibration
    assert not item.fast_sensor
    assert item.start is 1696171376701856700
    assert item.stop is 1696171386701856700
    assert item.stiffness is ref_passive_fixed_diode_with_height["kappa (pN/nm)"]
    assert item.force_sensitivity is ref_passive_fixed_diode_with_height["Rf (pN/V)"]
    assert item.displacement_sensitivity is ref_passive_fixed_diode_with_height["Rd (um/V)"]

    compare_to_reference_dict(item.power_spectrum_params(), test_name="power")
    compare_to_reference_dict(item._model_params(), test_name="model")
    compare_to_reference_dict(item.calibration_params(), test_name="calibration")

    # Validate that these work!
    dummy_voltage, _ = calibration_data
    calculate_power_spectrum(dummy_voltage, **item.power_spectrum_params())
    calibrate_force(dummy_voltage, **item.calibration_params())


def test_active_item_fixed_diode(compare_to_reference_dict, calibration_data):
    item = ForceCalibrationItem(ref_active)
    assert item.excluded_ranges == [(12.0, 22.0), (205.0, 265.0)]
    assert item.fit_range == (100.0, 23000.0)
    assert item.num_points_per_block == 200
    assert item.sample_rate == 78125
    assert item.active_calibration  # It is an active item!
    assert not item.fast_sensor
    assert item.start is 1713785826919398000
    assert item.stop is 1713785836900152600
    assert item.stiffness is ref_active["kappa (pN/nm)"]
    assert item.force_sensitivity is ref_active["Rf (pN/V)"]
    assert item.displacement_sensitivity is ref_active["Rd (um/V)"]

    compare_to_reference_dict(item.power_spectrum_params(), test_name="power_ac")
    compare_to_reference_dict(item._model_params(), test_name="model_ac")
    compare_to_reference_dict(item.calibration_params(), test_name="calibration_ac")

    # Validate that these work!
    dummy_voltage, nano = calibration_data
    calculate_power_spectrum(dummy_voltage, **item.power_spectrum_params())
    calibrate_force(dummy_voltage, driving_data=nano, **item.calibration_params())


def test_axial_fast_sensor(compare_to_reference_dict, calibration_data):
    item = ForceCalibrationItem(ref_axial)
    assert not item.active_calibration
    assert item.fast_sensor
    assert item.stiffness is ref_axial["kappa (pN/nm)"]
    assert item.force_sensitivity is ref_axial["Rf (pN/V)"]
    assert item.displacement_sensitivity is ref_axial["Rd (um/V)"]

    compare_to_reference_dict(item.power_spectrum_params(), test_name="power_axial")
    compare_to_reference_dict(item._model_params(), test_name="model_axial")
    compare_to_reference_dict(item.calibration_params(), test_name="calibration_axial")

    # Validate that these work!
    dummy_voltage, _ = calibration_data
    calculate_power_spectrum(dummy_voltage, **item.power_spectrum_params())
    calibrate_force(dummy_voltage, **item.calibration_params())


def test_non_full():
    item = ForceCalibrationItem(
        {
            "Kind": "Discard all calibration data",
            "Start time (ns)": 1714391268938540100,
            "Stop time (ns)": 1714391268938540200,
            "Response (pN/V)": 1.0,
        }
    )
    assert not item.stiffness
    assert not item.displacement_sensitivity
    assert item.force_sensitivity is 1.0
    assert item.start is 1714391268938540100
    assert item.stop is 1714391268938540200

    # To be on the safe side, we only allow extracting these fields for a full calibration.
    for field in (
        "fit_range",
        "num_points_per_block",
        "sample_rate",
        "excluded_ranges",
        "fast_sensor",
        "active_calibration",
    ):
        with pytest.raises(
            ValueError, match="These parameters are only available for a full calibration"
        ):
            getattr(item, field)

    for func in ("calibration_params", "power_spectrum_params"):
        with pytest.raises(
            ValueError, match="These parameters are only available for a full calibration"
        ):
            getattr(item, func)()
