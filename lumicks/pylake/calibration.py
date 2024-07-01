import re
from functools import wraps
from collections import UserDict

from tabulate import tabulate

from lumicks.pylake.force_calibration.power_spectrum_calibration import CalibrationPropertiesMixin


class ForceCalibrationItem(UserDict, CalibrationPropertiesMixin):
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
    def _sensor_type(self):
        if self.fast_sensor:
            return "fast_sensor"

        if "f_diode (Hz)" in self.data:
            return "standard sensor (fitted)"
        elif "Diode frequency (Hz)" in self.data:
            return "characterized standard sensor"
        else:
            return "unknown sensor"

    @property
    def kind(self):
        kind = self.data.get("Kind", "Unknown")
        if kind == "Full calibration":
            return "Active calibration" if self.active_calibration else "Passive calibration"
        else:
            return kind

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
        return self._repr_properties

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
            rf_ratio = less_blocking.force_sensitivity / previous_calibration.force_sensitivity
            recalibrated_force1x = f.force1x * rf_ratio
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
    def active_calibration(self):
        """Returns whether it was an active calibration or not

        Calibrations based on active calibration are less sensitive to assumptions about the
        bead diameter, viscosity, distance of the bead to the flow cell surface and temperature.
        During active calibration, the trap or nano-stage is oscillated sinusoidally. These
        oscillations result in a driving peak in the force spectrum. Using power spectral analysis,
        the force can then be calibrated without prior knowledge of the drag coefficient.

        While this results in improved accuracy, it may lead to an increase in the variability of
        results.

        .. note::

            When active calibration is performed using two beads, correction factors must be
            computed which account for the reduced flow field that forms around the beads due to
            the presence of a second bead.
        """
        return self.data.get("driving_frequency (Hz)") is not None

    @property
    def num_points_per_block(self):
        """Number of points per block used for spectral down-sampling"""
        if "Points per block" in self.data:
            return int(self.data["Points per block"])  # BL returns float which API doesn't accept

    @property
    def start(self):
        """Starting timestamp of this calibration

        Examples
        --------
        ::

            import lumicks.pylake as lk

            f = lk.File("file.h5")
            item = f.force1x.calibration[1]  # Grab a calibration item for force 1x

            # Slice the data corresponding to this item
            calibration_data = f.force1x[item.start : item.stop]

            # or alternatively:
            calibration_data = f.force1x[item]
        """
        return self.data.get("Start time (ns)")

    @property
    def stop(self):
        return self.data.get("Stop time (ns)")


def _filter_calibration(time_field, items, start, stop):
    """filter calibration data based on time stamp range [ns]"""
    if len(items) == 0:
        return []

    def timestamp(x):
        return x[time_field]

    items = sorted(items, key=timestamp)

    calibration_items = [x for x in items if start < timestamp(x) < stop]
    pre = [x for x in items if timestamp(x) <= start]
    if pre:
        calibration_items.insert(0, pre[-1])

    return calibration_items


class ForceCalibration:
    """Calibration handling

    Examples
    --------
    ::

        import lumicks.pylake as lk

        f = lk.File("passive_calibration.h5")
        print(f.force1x.calibration)  # Show force calibration items available

        calibration = f.force1x.calibration[1]  # Grab a calibration item for force 1x
    """

    def __init__(self, time_field, items, slice_start=None, slice_stop=None):
        """Calibration item

        Parameters
        ----------
        time_field : string
            name of the field used for time
        items : list[ForceCalibrationItem]
            list of force calibration items
        slice_start, slice_stop : int
            Start and stop index of the slice associated with these items
        """
        self._time_field = time_field
        self._src = items
        self._slice_start = slice_start
        self._slice_stop = slice_stop

    def _with_src(self, _src):
        return ForceCalibration(self._time_field, _src)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._with_src(self._src[item])

        return self._src[item]

    def __len__(self):
        return len(self._src)

    def __iter__(self):
        return iter(self._src)

    def __eq__(self, other):
        if not self._src or not other._src:
            return False

        return self._src == other._src

    def filter_calibration(self, start, stop):
        """Filter calibration based on time stamp range

        Parameters
        ----------
        start : int
            time stamp at start [ns]
        stop  : int
            time stamp at stop [ns]"""
        return ForceCalibration(
            self._time_field,
            _filter_calibration(self._time_field, self._src, start, stop),
            start,
            stop,
        )

    @staticmethod
    def from_field(hdf5, force_channel, time_field="Stop time (ns)") -> "ForceCalibration":
        """Fetch force calibration data from the HDF5 file

        Parameters
        ----------
        hdf5 : h5py.File
            A Bluelake HDF5 file.
        force_channel : str
            Calibration field to access (e.g. "Force 1x").
        time_field : str
            Attribute which holds the timestamp of the item (e.g. "Stop time (ns)").
        """

        if "Calibration" not in hdf5.keys():
            return ForceCalibration(time_field=time_field, items=[])

        items = []
        for calibration_item in hdf5["Calibration"].values():
            if force_channel in calibration_item:
                attrs = calibration_item[force_channel].attrs
                if time_field in attrs.keys():
                    items.append(ForceCalibrationItem(attrs))

        return ForceCalibration(time_field=time_field, items=items)

    def print_summary(self, tablefmt):
        return tabulate(
            (
                (
                    idx,
                    item.kind,
                    f"{item.stiffness:.2f}" if item.stiffness else "N/A",
                    f"{item.force_sensitivity:.2f}" if item.force_sensitivity else "N/A",
                    f"{item.displacement_sensitivity:.2f}" if item.force_sensitivity else "N/A",
                    item.hydrodynamically_correct,
                    item.distance_to_surface is not None,
                    bool(
                        self._slice_start
                        and (item.start >= self._slice_start)
                        and self._slice_stop
                        and (item.stop <= self._slice_stop)
                    ),
                )
                for idx, item in enumerate(self._src)
            ),
            tablefmt=tablefmt,
            headers=(
                "Index",
                "Kind",
                "Stiffness (pN/nm)",
                "Force sens. (pN/V)",
                "Disp. sens. (µm/V)",
                "Hydro",
                "Surface",
                "Data?",
            ),
        )

    def _repr_html_(self):
        return self.print_summary(tablefmt="html")

    def __str__(self):
        return self.print_summary(tablefmt="text")

    @staticmethod
    def from_dataset(hdf5, n, xy, time_field="Stop time (ns)") -> "ForceCalibration":
        """Fetch the force calibration data from the HDF5 file

        Parameters
        ----------
        hdf5 : h5py.File
            A Bluelake HDF5 file.
        n : int
            Trap index.
        xy : str
            Force axis (e.g. "x").
        time_field : str
            Attribute which holds the timestamp of the item (e.g. "Stop time (ns)").
        """

        if xy:
            return ForceCalibration.from_field(
                hdf5, force_channel=f"Force {n}{xy}", time_field=time_field
            )
        else:
            raise NotImplementedError(
                "Calibration is currently only implemented for single axis data"
            )
