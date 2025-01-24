import re
import copy
from functools import wraps
from collections import UserDict

from lumicks.pylake.channel import empty_slice
from lumicks.pylake.force_calibration.convenience import calibrate_force
from lumicks.pylake.force_calibration.calibration_models import DiodeCalibrationModel
from lumicks.pylake.force_calibration.detail.calibration_properties import (
    CalibrationPropertiesMixin,
)


class ForceCalibrationItem(UserDict, CalibrationPropertiesMixin):
    def __init__(
        self,
        dictionary=None,
        *,
        voltage=None,
        sum_voltage=None,
        driving=None,
        force_slice=None,
    ):
        super().__init__(dictionary)
        self._voltage = voltage
        self._sum_voltage = sum_voltage
        self._driving = driving
        self._force_slice = force_slice

    @property
    def voltage(self):
        """Uncalibrated voltage reading on the detector"""
        if self._voltage is not None:
            return self._voltage
        elif (
            self._force_slice
            and self._force_slice.start <= self.start
            and self._force_slice.stop >= self.stop
        ):
            force_slice = self._force_slice[self.start : self.stop]
            if not force_slice.calibration:
                return empty_slice

            ref_calibration = force_slice.calibration[0]
            return (force_slice - ref_calibration.offset) / ref_calibration.force_sensitivity

        return empty_slice

    @property
    def sum_voltage(self):
        """Uncalibrated sum voltage on the detector"""
        return self._sum_voltage if self._sum_voltage is not None else empty_slice

    @property
    def driving(self):
        """Driving signal used for active calibration"""
        return self._driving if self._driving is not None else empty_slice

    @staticmethod
    def _verify_full(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if (kind := self.data.get("Kind", "Unknown")) != "Full calibration":
                raise ValueError(
                    "These parameters are only available for a full calibration. Instead, this is "
                    f"a calibration item where the following operation was applied: {kind}."
                )

            return method(self, *args, **kwargs)

        return wrapper

    def _get_parameter(self, _, bluelake_key):
        """Grab a parameter"""
        if bluelake_key in self:
            return self[bluelake_key]

    @property
    def _fitted_diode(self):
        """Diode parameters were fitted"""
        return "f_diode (Hz)" in self or "alpha" in self

    @property
    def diode_calibration(self) -> DiodeCalibrationModel | None:
        """Diode calibration model

        The detector used to measure forces has a limited bandwidth. This bandwidth is typically
        characterized by two parameters, a diode frequency and relaxation factor (which gives the
        fraction of light that is instantaneously transmitted).

        Some force detection diodes have been pre-calibrated in the factory. For these sensors,
        this property will return a model that will allow you to calculate the diode parameters
        at a particular trap sum power.

        Examples
        --------
        ::

            import lumicks.pylake as lk

            f = lk.File("passive_calibration.h5")
            item = f.force1x.calibration[0]  # Grab a calibration item for force 1x
            diode_model = item.diode_calibration  # Grab diode model
            diode_parameters = diode_model(item.trap_power)  # Grab diode parameters

            # Verify that the calibration parameters at this trap power are the same as we found
            # in the item in the first place.
            assert diode_parameters["fixed_diode"] == item.diode_frequency
            assert diode_parameters["fixed_alpha"] == item.diode_relaxation_factor
        """
        try:
            return DiodeCalibrationModel.from_calibration_dict(self.data)
        except ValueError:
            return None  # No diode calibration present

    @property
    def trap_power(self):
        """Average trap sum power in volts during calibration

        Note that this property is only available for full calibrations with calibrated diodes."""
        return self.data.get("Trap sum power (V)")

    @property
    def _sensor_type(self):
        if self.fast_sensor:
            return "fast_sensor"

        if "f_diode (Hz)" in self.data:
            return "standard sensor (fitted)"
        elif "Diode frequency (Hz)" in self.data:
            return "characterized standard sensor"
        else:
            return "unknown sensor"

    @_verify_full
    def power_spectrum_params(self):
        """Returns parameters with which the power spectrum was calculated

        Examples
        --------
        ::

            import lumicks.pylake as lk

            f = lk.File("passive_calibration.h5")
            calibration = f.force1x.calibration[1]  # Grab a calibration item for force 1x

            # Slice the data corresponding to the calibration we want to reproduce.
            calib_slice = f.force1x[calibration]

            # De-calibrate to volts using the calibration that was active before this slice.
            previous_calibration = calib_slice.calibration[0]
            calib_slice = calib_slice / previous_calibration.force_sensitivity

            power_spectrum_params = previous_calibration.power_spectrum_params()
            power_spectrum = lk.calculate_power_spectrum(calib_slice.data, **power_spectrum_params)
            power_spectrum.plot()
        """
        return {
            "num_points_per_block": self.num_points_per_block,
            "sample_rate": self.sample_rate,
            "excluded_ranges": self.excluded_ranges,
            "fit_range": self.fit_range,
        }

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            return getattr(self, item)

    def __repr__(self):
        """Returns properties in a dataclass-style formatting"""
        properties = ", ".join(
            (f"{prop}={str(getattr(self, prop))}" for prop in self._results + self._parameters)
        )
        return f"{self.__class__.__name__}({properties})"

    def _check_has_data(self):
        if not len(self.voltage):
            raise ValueError(
                "This calibration item does not contain the raw data. If you still have the "
                "timeline force data, you can de-calibrate that and perform the re-calibration"
                "manually. See the pylake tutorial on force calibration for more information."
            )

    def plot(self):
        self._check_has_data()
        self.recalibrate_with().plot()

    def plot_spectrum_residual(self):
        """Plot the residuals of the fitted spectrum.

        This diagnostic plot can be used to determine how well the spectrum fits the data. While
        it cannot be used to diagnose over-fitting (being unable to reliably estimate parameters
        due to insufficient information in the data), it can be used to diagnose under-fitting (the
        model not fitting the data adequately).

        In an ideal situation, the residual plot should show a noise band around 1 without any
        systematic deviations.
        """
        self._check_has_data()
        self.recalibrate_with().plot_spectrum_residual()

    def recalibrate_with(self, **params):
        """Returns a calibration structure with some parameters overridden.

        For a full list of parameters to override, please see
        :func:`~lumicks.pylake.calibrate_force()`"""
        self._check_has_data()
        active_data = {"driving_data": self.driving.data} if self.active_calibration else {}
        return calibrate_force(
            self.voltage.data, **(self.calibration_params() | active_data | params)
        )

    @_verify_full
    def _model_params(self):
        """Returns parameters with which to create an active or passive calibration model"""

        # TODO: Model needs to support fixed_diode and fixed_alpha, keep API private until finalized
        params = {
            "bead_diameter": self.bead_diameter,
            "rho_bead": self.rho_bead,
            "rho_sample": self.rho_sample,
            "viscosity": self.viscosity,
            "temperature": self.temperature,
            "driving_frequency_guess": self.driving_frequency_guess,
            "distance_to_surface": self.distance_to_surface,
            # Fixed diode parameters (only available when diode was fixed)
            "fixed_alpha": self.diode_relaxation_factor if not self.fitted_diode else None,
            "fixed_diode": self.diode_frequency if not self.fitted_diode else None,
            # Only available for axial with active
            "drag": self.transferred_lateral_drag_coefficient,
            # Other properties
            "sample_rate": self.sample_rate,
            "fast_sensor": self.fast_sensor,
            "axial": bool(self.data.get("Axial calibration")),
            "hydrodynamically_correct": self.hydrodynamically_correct,
        }

        return {key: value for key, value in params.items() if value is not None}

    @_verify_full
    def calibration_params(self):
        """Returns parameters to calculate the same calibration as the one performed by Bluelake

        Examples
        --------
        ::

            import lumicks.pylake as lk

            f = lk.File("passive_calibration.h5")
            calibration = f.force1x.calibration[1]  # Grab a calibration item for force 1x

            # Slice the data corresponding to the calibration we want to reproduce.
            calib_slice = f.force1x[calibration]

            # De-calibrate to volts using the calibration that was active before this slice.
            previous_calibration = calib_slice.calibration[0]
            calib_slice = calib_slice / previous_calibration.force_sensitivity

            calibration_params = previous_calibration.calibration_params()
            new_calibration = lk.calibrate_force(calib_slice.data, **calibration_params)
            new_calibration.plot()

            # Make a new calibration, but change the amount of blocking
            less_blocking_params = calibration_params | {"num_points_per_block": 200}
            less_blocking = lk.calibrate_force(calib_slice.data, **less_blocking_params)
            less_blocking.plot()

            # Recalibrate the force channels
            recalibrated_force1x = f.force1x.recalibrate_force(less_blocking)
        """
        return (
            self.power_spectrum_params()
            | self._model_params()
            | {"active_calibration": self.active_calibration}
        )

    @property
    def _excluded_ranges(self):
        """Returns the frequency exclusion ranges."""
        exclusion_range_indices = [
            int(match[0])
            for key in self.data.keys()
            if (match := re.findall(r"Exclusion range ([\d+]) \(min\.\) \(Hz\)", key))
        ]
        return [
            (
                self.data[f"Exclusion range {exclusion_idx} (min.) (Hz)"],
                self.data[f"Exclusion range {exclusion_idx} (max.) (Hz)"],
            )
            for exclusion_idx in sorted(exclusion_range_indices)
        ]

    @property
    def _fit_range(self):
        """Returns the spectral fit range used for the calibration"""
        if "Fit range (min.) (Hz)" in self.data:
            return (
                self.data["Fit range (min.) (Hz)"],
                self.data["Fit range (max.) (Hz)"],
            )

    def _with_timestamp(self, applied_timestamp):
        """Return a copy of this item with a timestamp of when it was applied"""
        item = copy.copy(self)
        item.data = copy.deepcopy(self.data)
        item.data["Timestamp (ns)"] = applied_timestamp
        return item

    @property
    def sample_rate(self):
        """Returns the data sample rate"""
        if "Sample rate (Hz)" in self.data:
            return int(self.data["Sample rate (Hz)"])

    @property
    def number_of_samples(self):
        """Number of fitted samples (-)."""
        return self.data.get("Number of samples")

    @property
    def num_points_per_block(self):
        """Number of points per block used for spectral down-sampling"""
        if "Points per block" in self.data:
            return int(self.data["Points per block"])  # BL returns float which API doesn't accept
