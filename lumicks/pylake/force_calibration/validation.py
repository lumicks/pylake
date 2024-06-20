import numpy as np


def check_hydro_enabled(calibration):
    """Check whether users are following the recommended hydro settings

    Parameters
    ----------
    calibration : lk.force_calibration.power_spectrum_calibration.CalibrationResults
        Power spectrum calibration result.
    """
    if (diameter := calibration.params["Bead diameter"].value) > 1.5:
        criterion = np.inf
        if (distance := calibration.params.get("Distance to surface")) and distance.value:
            criterion = distance.value / (diameter / 2)

        if criterion > 1.5:
            if not calibration.model.hydrodynamically_correct and not calibration.model.axial:
                return (
                    "Using the hydrodynamically correct model will lead to more accurate force "
                    "calibrations. It is generally recommended for bead radii > 1.5 microns as "
                    "long as you are not calibrating closer than 1.5 times the bead radius from "
                    "the surface."
                )


def check_corner_frequency(calibration):
    """Checks whether the corner frequency is below the spectral resolution or fit range

    Parameters
    ----------
    calibration : lk.force_calibration.power_spectrum_calibration.CalibrationResults
        Power spectrum calibration result.
    """
    fc = calibration.results["fc"].value
    if fc < (freq_resolution := calibration.ps_data.frequency_bin_width):
        return (
            f"Corner frequency ({fc:.0f} Hz) is below the spectral frequency resolution "
            f"({freq_resolution:.0f} Hz). Consider increasing the acquisition duration or reducing "
            "the number of points per block."
        )

    if fc < (min_freq := calibration.ps_data.frequency[0]):
        return (
            f"Estimated corner frequency ({fc:.0f} Hz) is below lowest frequency in the power "
            f"spectrum ({min_freq:.0f} Hz). Consider lowering the minimum fit range."
        )


def upsample_frequency_axis(ps):
    """Determines what frequencies would have contributed to a spectrum with a specific
    blocking."""
    df, num = ps.frequency_bin_width, ps.num_points_per_block
    upsampled_freq = ps.frequency[:, np.newaxis] + (np.arange(-num / 2, num / 2) + 0.5) * df / num
    return upsampled_freq


def check_blocking_error(calibration, blocking_threshold):
    """Check whether we are not over-averaging the power spectral density.

    Excessive blocking can lead to over-smoothing of the spectrum. The error from blocking is
    proportional to the relative change in the power spectrum across a block. This error is
    on the order of `1/fc` and therefore typically small. For low corner frequencies (such as
    those encountered in axial force detection) it can become relevant.

    Parameters
    ----------
    calibration : lk.force_calibration.power_spectrum_calibration.CalibrationResults
        Power spectrum calibration result.
    blocking_threshold : float
        How many percent can a single spectral point be off when comparing blocked vs non-blocked.
    """

    def calculate_model_blocking_error():
        """Calculates how much error the blocking induces in the spectrum"""
        blocked = np.mean(calibration(upsample_frequency_axis(calibration.ps_model)), axis=1)
        non_blocked = calibration(calibration.ps_model.frequency)
        error = 100 * abs(blocked - non_blocked) / non_blocked
        return error

    if (blocking_error := np.max(calculate_model_blocking_error())) > blocking_threshold:
        return (
            "Maximum spectral error exceeds threshold."
            f"Spectral error: {blocking_error:.3f} > {blocking_threshold:.3f}."
            "This likely means blocking was applied too aggressively. Low corner frequencies "
            "require lower blocking factors."
        )


def check_calibration_factor_precision(calibration, factor=0.2):
    """Verify that the corner frequency, displacement sensitivity and stiffness have been precisely
    estimated.

    Parameters
    ----------
    calibration : lk.force_calibration.power_spectrum_calibration.CalibrationResults
        Power spectrum calibration result.
    factor : float
        Factor which determines how large the standard error may be (std_err <= factor * parameter).
    """

    for parameter, description in (
        ("fc", "corner frequency"),
        ("Rd", "displacement sensitivity"),
        ("kappa", "stiffness"),
    ):
        if (
            calibration.results[f"err_{parameter}"].value
            > factor * calibration.results[parameter].value
        ):
            return (
                f"More than {factor:.0%} error in the {description}. There is high uncertainty in "
                "the calibration factors."
            )


def check_diode_identifiability(calibration, alpha):
    """Check whether the diode model and corner frequency have become non-identifiable.

    If we fitted a diode model, check that the diode parameter does not fall into the
    approximate confidence interval of the corner frequency. Note that checking for
    overlapping confidence intervals is problematic, since the diode confidence interval
    can be large when it is difficult to estimate (but this is not necessarily a problem).

    Parameters
    ----------
    calibration : lk.force_calibration.power_spectrum_calibration.CalibrationResults
        Power spectrum calibration result.
    alpha : float
        Confidence level for testing whether the diode frequency falls inside the confidence
        interval of the corner frequency.
    """
    from scipy import stats

    fc = calibration.results["fc"].value
    if "err_f_diode" in calibration.results:
        fc_width = stats.norm.ppf(1 - alpha / 2, scale=calibration.results["err_fc"].value)
        if fc - fc_width <= calibration.results["f_diode"].value <= fc + fc_width:
            return (
                "Warning, the estimate for the parasitic filtering frequency falls within the "
                "confidence interval for the corner frequency. This means that the corner "
                "frequency may be too high to be reliably estimated. Lower the laser power or "
                "use a pre-calibrated diode model for more reliable results."
            )


def check_backing(calibration, desired_backing):
    """Measures how good the fit is to the data.

    Note that this criterion measures how unlikely the data is under the current model and is very
    sensitive to violations of this model. Including noise peaks or having an under-fitting model
    trigger it. That in itself doesn't necessarily mean the calibration factors obtained are off
    by much; just that sum of squared residuals are different from expected.

    Parameters
    ----------
    calibration : lk.force_calibration.power_spectrum_calibration.CalibrationResults
        Power spectrum calibration result.
    desired_backing : float
        Desired level of backing for the spectral fit. A backing of 1% means that if this
        test fires, there's a 1% chance that we erroneously produced an error.
    """
    if (backing := calibration.results["backing"].value) < desired_backing:
        return (
            f"Statistical backing is low ({backing:.3e} < {desired_backing:.3e}). It is "
            "possible that the fit of the thermal calibration spectrum is bad. Verify that "
            "your fit does not show a residual trend with `result.plot_spectrum_residual()`."
        )


def validate_results(calibration, alpha=0.1, desired_backing=1e-12, blocking_threshold=3):
    """This function checks some force calibration quality criteria.

    Parameters
    ----------
    calibration : lk.force_calibration.power_spectrum_calibration.CalibrationResults
        Power spectrum calibration result.
    alpha : float
        Confidence level for testing whether the diode frequency falls inside the confidence
        interval of the corner frequency.
    desired_backing : float
        Desired level of backing for the spectral fit. A backing of 1% means that if this
        test fires, there's a 1% chance that we erroneously produced an error. Note that this
        criterion measures how unlikely the data is under the current model and is very
        sensitive to noise peaks and model mis-specification.
    blocking_threshold : float
        How many percent can a single spectral point be off when comparing blocked vs non-blocked.
    """
    tests = [
        (check_hydro_enabled, {}),
        (check_corner_frequency, {}),
        (check_diode_identifiability, {"alpha": alpha}),
        (check_blocking_error, {"blocking_threshold": blocking_threshold}),
        (check_calibration_factor_precision, {}),
        (check_backing, {"desired_backing": desired_backing}),
    ]

    for test, params in tests:
        if result := test(calibration, **params):
            return result
