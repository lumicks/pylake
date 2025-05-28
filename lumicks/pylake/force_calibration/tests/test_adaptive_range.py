import os
from copy import deepcopy

import numpy as np
import pytest

import lumicks.pylake.force_calibration.power_spectrum_calibration as psc
from lumicks.pylake.force_calibration.power_spectrum import PowerSpectrum
from lumicks.pylake.force_calibration.calibration_models import (
    FixedDiodeModel,
    PassiveCalibrationModel,
)


def test_corner_case_minimum():
    """For really low corner frequencies, the upper limit of the fitting range can end up below
    the lowest frequency. We have an explicit extra condition that the upper limit of the
    fitting range can never be lower than a factor times the lower bound of the fit range. This
    condition is tested with actual data here."""
    ps = PowerSpectrum(
        **np.load(os.path.join(os.path.dirname(__file__), "data/big_bead_low_fc.npz"))
    )
    model = PassiveCalibrationModel(10.1, temperature=25, hydrodynamically_correct=True)
    model._filter = FixedDiodeModel(15700.12316662277, 0.5234935370575797)
    fit = psc.fit_power_spectrum(ps, model, corner_frequency_factor=4)
    np.testing.assert_allclose(fit.corner_frequency, 308.0, rtol=1e-3)
    np.testing.assert_equal(fit["Number of iterations"].value, 3)


def test_noise_floor(reference_calibration_result):
    ps_calibration, model, reference_spectrum = reference_calibration_result
    bad_spectrum = reference_spectrum.power.copy()
    bad_spectrum += 1e-3 * bad_spectrum[1]  # Add offset
    bad_spectrum = reference_spectrum.with_spectrum(
        bad_spectrum, num_points_per_block=reference_spectrum.num_points_per_block
    )

    model_fixed_diode = deepcopy(model)
    model_fixed_diode._filter = FixedDiodeModel(
        diode_frequency=ps_calibration.diode_frequency,
        diode_alpha=ps_calibration.diode_relaxation_factor,
    )
    bad_calibration = psc.fit_power_spectrum(
        power_spectrum=bad_spectrum, model=model_fixed_diode, loss_function="gaussian"
    )
    with pytest.warns(
        RuntimeWarning,
        match="Fitting with adaptive fitting ranges may be unreliable when fitting the "
        "parasitic filtering parameters",
    ):
        warned = psc.fit_power_spectrum(
            power_spectrum=bad_spectrum,
            model=model,
            loss_function="gaussian",
            corner_frequency_factor=4,
        )

    good_fit = psc.fit_power_spectrum(
        power_spectrum=bad_spectrum,
        model=model_fixed_diode,
        loss_function="gaussian",
        corner_frequency_factor=4,
    )

    results = {
        "D": 0.0018512505734895896,
        "Rd": 7.253677199344564,
        "Rf": 1243.966729922322,
        "kappa": 0.17149463585651784,
    }

    # Results with the mitigation should be within 5% of true
    for name, expected_result in results.items():
        np.testing.assert_allclose(good_fit[name].value, expected_result, rtol=5e-2)
        np.testing.assert_allclose(warned[name].value, expected_result, rtol=3e-1)  # oof tolerance
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(bad_calibration[name].value, expected_result, rtol=5e-2)

    np.testing.assert_allclose(good_fit["Number of iterations"].value, 2)
    np.testing.assert_allclose(bad_calibration["Number of iterations"].value, 1)
    np.testing.assert_allclose(warned["Number of iterations"].value, 3)
