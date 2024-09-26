import numpy as np

from lumicks.pylake.force_calibration.detail.calibration_properties import (
    CalibrationPropertiesMixin,
)


class CalibrationResults(CalibrationPropertiesMixin):
    """Power spectrum calibration results.

    Attributes
    ----------
    model : PassiveCalibrationModel | ActiveCalibrationModel
        Model used for calibration.
    ps_model : PowerSpectrum
        Power spectrum of the fitted model.
    ps_data : PowerSpectrum
        Power spectrum of the data that the model was fitted to.
    params : dict
        Dictionary of input parameters.
    results : dict
        Dictionary of calibration results.
    fitted_params : np.ndarray
        Fitted parameters.
    """

    def __init__(self, model, ps_model, ps_data, params, results, fitted_params):
        self.model = model
        self.ps_model = ps_model
        self.ps_data = ps_data
        self.params = params
        self.results = results
        self.fitted_params = fitted_params

        # A few parameters have to be present for this calibration to be used.
        mandatory_params = ["kappa", "Rf"]
        for key in mandatory_params:
            if key not in results:
                raise RuntimeError(f"Calibration did not provide calibration parameter {key}")

    def __call__(self, frequency):
        """Evaluate the spectral model for one or more frequencies

        Parameters
        ----------
        frequency : array_like
            One or more frequencies at which to evaluate the spectral model.
        """
        return self.model(frequency, *self.fitted_params)

    def __contains__(self, key):
        return key in self.params or key in self.results

    def __getitem__(self, item):
        return self.params[item] if item in self.params else self.results[item]

    def __repr__(self):
        """Returns properties in a dataclass-style formatting"""
        properties = ", ".join(
            (f"{prop}={getattr(self, prop)}" for prop in self._results + self._parameters)
        )
        return f"{self.__class__.__name__}({properties})"

    @property
    def _fit_range(self):
        """Fitted range"""
        return self.ps_data._fit_range

    @property
    def _excluded_ranges(self):
        """Frequency exclusion ranges"""
        return self.ps_data._excluded_ranges

    def _get_parameter(self, pylake_key, _):
        """Grab a parameter"""
        if pylake_key in self:
            return self[pylake_key].value

    @property
    def _fitted_diode(self):
        """Diode parameters were fitted"""
        return "f_diode" in self.results or "alpha" in self.results

    @property
    def sample_rate(self):
        """Acquisition sample rate (Hz)."""
        return self.ps_data.sample_rate

    @property
    def number_of_samples(self):
        """Number of fitted samples (-)."""
        return self.ps_data.total_sampled_used

    def plot(self, show_excluded=False, show_active_peak=False):
        """Plot the fitted spectrum

        Parameters
        ----------
        show_excluded : bool
            Show fitting regions excluded from the fit
        show_active_peak : bool
            Show active calibration peak when available

        Raises
        ------
        ValueError
            If specifying `show_active_peak=True` for a passive calibration.
        """
        import matplotlib.pyplot as plt

        if show_active_peak and not hasattr(self.model, "output_power"):
            raise ValueError(
                "Requested to plot an active calibration peak while this is not an active "
                "calibration. Please specify `show_active_peak=False` when plotting passive"
                "calibration results."
            )

        self.ps_data.plot(label="Data", show_excluded=show_excluded)
        self.ps_model.plot(label="Model")

        if show_active_peak:
            self.model.output_power.ps.plot(label="Active peak")

        plt.legend()

    def plot_spectrum_residual(self):
        """Plot the residuals of the fitted spectrum.

        This diagnostic plot can be used to determine how well the spectrum fits the data. While
        it cannot be used to diagnose over-fitting (being unable to reliably estimate parameters
        due to insufficient information in the data), it can be used to diagnose under-fitting (the
        model not fitting the data adequately).

        In an ideal situation, the residual plot should show a noise band around 1 without any
        systematic deviations.
        """
        import matplotlib.pyplot as plt

        residual = self.ps_data.power / self.ps_model.power
        theoretical_std = 1.0 / np.sqrt(self.ps_model.num_points_per_block)

        plt.plot(self.ps_data.frequency, residual, ".")
        plt.axhline(1.0 + theoretical_std, color="k", linestyle="--")
        plt.axhline(1.0 - theoretical_std, color="k", linestyle="--")
        plt.ylabel("Data / Fit [-]")
        plt.xlabel("Frequency [Hz]")
