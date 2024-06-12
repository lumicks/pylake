from collections import UserDict
import re

CALIBRATION_PARAM_MAPPING = {
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
    "drag": "gamma_ex_lateral",
}


def _read_from_bl_dict(bl_dict, mapping):
    return {key_param: bl_dict.get(key_bl) for key_param, key_bl in mapping.items()}


class ForceCalibrationItem(UserDict):
    def _verify_full(self):
        if (kind := self.data.get("Kind", "Unknown")) != "Full calibration":
            raise ValueError(
                "These parameters are only available for a full calibration. Instead, this is a "
                f"calibration item where the following operation was applied: {kind}."
            )

    def power_spectrum_params(self):
        self._verify_full()

        return {
            "num_points_per_block": self.num_points_per_block,
            "sample_rate": self.sample_rate,
            "excluded_ranges": self.excluded_ranges,
            "fit_range": self.fit_range,
        }

    def model_params(self):
        self._verify_full()
        model_params = _read_from_bl_dict(self.data, CALIBRATION_PARAM_MAPPING)
        model_params["fast_sensor"] = self.fast_sensor
        model_params["axial"] = bool(self.data.get("Axial calibration"))
        model_params["hydrodynamically_correct"] = bool(
            self.data.get("Hydrodynamic correction enabled")
        )
        return model_params

    @property
    def calibration_params(self):
        model_params = self.model_params()
        model_params["active_calibration"] = self.active_calibration

        def check_defined(pair):
            return pair[1] is not None

        return dict(filter(check_defined, (self.power_spectrum_params() | model_params).items()))

    @property
    def excluded_ranges(self):
        self._verify_full()

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
    def fit_range(self):
        self._verify_full()

        return (
            self.data["Fit range (min.) (Hz)"],
            self.data["Fit range (max.) (Hz)"],
        )

    @property
    def active_calibration(self):
        return self.data.get("driving_frequency (Hz)") is not None

    @property
    def fast_sensor(self):
        # If it is not a fixed diode or free diode, it's a fast sensor
        diode_fields = ("Diode frequency (Hz)", "f_diode (Hz)")
        return not any(f in self.data.keys() for f in diode_fields)

    @property
    def num_points_per_block(self):
        """Number of points per block used for spectral down-sampling"""
        return self.data.get("Points per block")

    @property
    def start(self):
        return self.data.get("Start time (ns)")

    @property
    def stop(self):
        return self.data.get("Stop time (ns)")

    @property
    def sample_rate(self):
        return self.data.get("Sample rate (Hz)")

    @property
    def stiffness(self):
        """Stiffness in pN/nm"""
        return self.data.get("kappa (pN/nm)")

    @property
    def force_sensitivity(self):
        """Force sensitivity in pN/V"""
        return self.data.get("Rf (pN/V)")

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
