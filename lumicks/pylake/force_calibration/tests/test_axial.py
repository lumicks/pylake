import numpy as np
import pytest

from lumicks.pylake.force_calibration.calibration_models import (
    ActiveCalibrationModel,
    PassiveCalibrationModel,
)
from lumicks.pylake.force_calibration.detail.drag_models import faxen_factor, brenner_axial
from lumicks.pylake.force_calibration.power_spectrum_calibration import (
    fit_power_spectrum,
    calculate_power_spectrum,
)
from lumicks.pylake.force_calibration.tests.data.simulate_calibration_data import (
    generate_active_calibration_test_data,
)


@pytest.mark.parametrize("hydro", [False, True])
def test_axial_calibration(reference_models, hydro):
    """We currently have no way to perform active axial calibration.

    However, we can make the passive calibration slightly more accurate by transferring the active
    calibration result from the lateral calibration over to it.

    This test performs an integration test which tests whether carrying over the drag coefficient
    from a lateral calibration to an axial one produces the correct results. To test this, we
    deliberately mis-specify our viscosity in the models (since active calibration is more
    robust against mis-specification of viscosity and bead radius).

    Approaching the surface leads to an increase in the drag coefficient:

        gamma(h) = gamma_bulk * correction_factor(h)

    For lateral this factor is given by a different function than axial. The experimental drag
    coefficient returned by the calibration procedure (gamma_ex) is the back-corrected bulk drag
    coefficient gamma_bulk. This can be directly transferred to the axial direction which then
    applies the correct forward correction factor for axial. This test verifies that this behaviour
    stays preserved (as we rely on it)."""
    np.random.seed(17256246)
    viscosity = 0.0011
    shared_pars = {"bead_diameter": 0.5, "temperature": 20}
    sim_params = {
        "sample_rate": 78125,
        "duration": 10,
        "stiffness": 0.05,
        "pos_response_um_volt": 0.6,
        "driving_sinusoid": (500, 31.9563),
        "diode": (1.0, 10000),
        **shared_pars,
    }
    # For reference, these parameters lead to a true bulk gamma of:
    gamma_ref = 5.183627878423158e-09

    # Distance to the surface
    dist = 1.5 * shared_pars["bead_diameter"] / 2

    def height_simulation(height_factor):
        """We hack in height dependence using the viscosity. Since the calibration procedure covers
        the same bead, we can do this (gamma_bulk is linearly proportional to the viscosity and
        bead size).

        height_factor : callable
            Provides the height dependent drag model.
        """
        return generate_active_calibration_test_data(
            **sim_params,
            viscosity=viscosity
            * height_factor(dist * 1e-6, shared_pars["bead_diameter"] * 1e-6 / 2),
        )

    volts_lateral, stage = height_simulation(faxen_factor)
    lateral_model = ActiveCalibrationModel(
        stage,
        volts_lateral,
        **shared_pars,
        sample_rate=sim_params["sample_rate"],
        viscosity=viscosity * 2,  # We mis-specify viscosity since we measure experimental drag
        driving_frequency_guess=32,
        hydrodynamically_correct=hydro,
        distance_to_surface=dist,
    )
    ps_lateral = calculate_power_spectrum(volts_lateral, sample_rate=78125)
    lateral_fit = fit_power_spectrum(ps_lateral, lateral_model)
    np.testing.assert_allclose(
        lateral_fit[lateral_model._measured_drag_fieldname].value, gamma_ref, rtol=5e-2
    )

    # Axial calibration
    axial_model = PassiveCalibrationModel(
        **shared_pars,
        viscosity=viscosity * 2,  # We deliberately mis-specify the viscosity to test the transfer
        hydrodynamically_correct=False,
        distance_to_surface=dist,
        axial=True,
    )

    # Transfer the result to axial calibration
    axial_model._set_drag(lateral_fit[lateral_model._measured_drag_fieldname].value)

    volts_axial, stage = height_simulation(brenner_axial)
    ps_axial = calculate_power_spectrum(volts_axial, sample_rate=78125)
    axial_fit = fit_power_spectrum(ps_axial, axial_model)
    np.testing.assert_allclose(axial_fit["gamma_ex_lateral"].value, gamma_ref, rtol=5e-2)
    np.testing.assert_allclose(axial_fit["kappa"].value, sim_params["stiffness"], rtol=5e-2)
    assert (
        axial_fit["gamma_ex_lateral"].description
        == "Bulk drag coefficient from lateral calibration"
    )
