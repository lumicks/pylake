import re
from functools import wraps
from collections import UserDict


def _read_from_bl_dict(bl_dict):
    """Returns a dictionary of calibration parameters from a calibration dictionary exported from
    Bluelake where the keys respond to parameter names used in Pylake.

    Parameters
    ----------
    bl_dict : dict[str, float]
        Raw dictionary coming from a calibration item.
    """

    mapping = {
        "bead_diameter": "Bead diameter (um)",
        "rho_bead": "Bead density (Kg/m3)",
        "rho_sample": "Fluid density (Kg/m3)",
        "viscosity": "Viscosity (Pa*s)",
        "temperature": "Temperature (C)",
        "driving_frequency_guess": "Driving data frequency (Hz)",
        "distance_to_surface": "Bead center height (um)",
        # Fixed diode parameters (only available when diode was fixed)
        "fixed_alpha": "Diode alpha",
        "fixed_diode": "Diode frequency (Hz)",
        # Only available for axial with active
        "drag": "gamma_ex_lateral (kg/s)",
    }

    return {
        key_param: bl_dict[key_bl] for key_param, key_bl in mapping.items() if key_bl in bl_dict
    }


class ForceCalibrationItem(UserDict):
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

    @_verify_full
    def _model_params(self):
        """Returns parameters with which to create an active or passive calibration model"""

        # TODO: Model needs to support fixed_diode and fixed_alpha, keep API private until finalized
        return _read_from_bl_dict(self.data) | {
            "sample_rate": self.sample_rate,
            "fast_sensor": self.fast_sensor,
            "axial": bool(self.data.get("Axial calibration")),
            "hydrodynamically_correct": bool(self.data.get("Hydrodynamic correction enabled")),
        }

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
        """
        return (
            self.power_spectrum_params()
            | self._model_params()
            | {"active_calibration": self.active_calibration}
        )

    @property
    @_verify_full
    def excluded_ranges(self):
        """Returns the frequency exclusion ranges

        When fitting power spectra, specific frequency intervals of the spectrum can be ignored to
        exclude noise peaks. This attribute returns a list of these intervals in Hertz."""
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
    @_verify_full
    def fit_range(self):
        """Returns the spectral fit range used for the calibration"""
        return (
            self.data["Fit range (min.) (Hz)"],
            self.data["Fit range (max.) (Hz)"],
        )

    @property
    @_verify_full
    def sample_rate(self):
        """Returns the data sample rate"""
        return self.data.get("Sample rate (Hz)")

    @property
    @_verify_full
    def active_calibration(self):
        """Returns whether it was an active calibration or not"""
        return self.data.get("driving_frequency (Hz)") is not None

    @property
    @_verify_full
    def fast_sensor(self):
        """Returns whether it was a fast sensor or not"""
        # If it is not a fixed diode or free diode, it's a fast sensor
        return not any(f in self.data.keys() for f in ("Diode frequency (Hz)", "f_diode (Hz)"))

    @property
    @_verify_full
    def num_points_per_block(self):
        """Number of points per block used for spectral down-sampling"""
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

    @property
    def stiffness(self):
        """Stiffness in pN/nm"""
        return self.data.get("kappa (pN/nm)")

    @property
    def force_sensitivity(self):
        """Force sensitivity in pN/V"""
        return self.data.get("Response (pN/V)")

    @property
    def displacement_sensitivity(self):
        """Displacement sensitivity in um/V"""
        return self.data.get("Rd (um/V)")


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

    Parameters
    ----------
    A source of calibration data

    Parameters
    ----------
    time_field : string
        name of the field used for time
    items : list
        list of dictionaries containing raw calibration attribute data
    """

    def __init__(self, time_field, items):
        self._time_field = time_field
        self._items = items

    def filter_calibration(self, start, stop):
        """Filter calibration based on time stamp range

        Parameters
        ----------
        start : int
            time stamp at start [ns]
        stop  : int
            time stamp at stop [ns]"""
        return _filter_calibration(self._time_field, self._items, start, stop)

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
