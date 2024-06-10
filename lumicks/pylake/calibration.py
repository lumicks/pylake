import re

CALIBRATION_PARAM_MAPPING = {
    "bead_diameter": ("Bead diameter (um)", None),
    "rho_bead": ("Bead density (Kg/m3)", None),
    "rho_sample": ("Fluid density (Kg/m3)", None),
    "axial": ("Axial calibration", False),
    "hydrodynamically_correct": ("Hydrodynamic correction enabled", False),
    "viscosity": ("Viscosity (Pa*s)", None),
    "temperature": ("Temperature (C)", None),
    "driving_frequency_guess": ("Driving data frequency (Hz)", None),
    "distance_to_surface": ("Bead center height (um)", None),
    # Fixed diode parameters (only available when diode was fixed)
    "fixed_alpha": ("Diode alpha", None),
    "fixed_diode": ("Diode frequency (Hz)", None),
    # Only available for axial with active
    "drag": ("gamma_ex_lateral", None),
}

POWER_SPECTRUM_MAPPING = {
    "num_points_per_block": ("Points per block", None),
    "sample_rate": ("Sample rate (Hz)", None),
}


def read_from_bl_dict(bl_dict, mapping):
    return {
        key_param: bl_dict.get(key_bl, default) for key_param, (key_bl, default) in mapping.items()
    }


def extract_used_params(calibration_item, omit_defaults=True):
    exclusion_range_indices = [
        int(match[0])
        for key in calibration_item.keys()
        if (match := re.findall(r"Exclusion range ([\d+]) \(min\.\) \(Hz\)", key))
    ]
    power_spectrum_params = read_from_bl_dict(calibration_item, POWER_SPECTRUM_MAPPING) | {
        "excluded_ranges": [
            (
                calibration_item[f"Exclusion range {exclusion_idx} (min.) (Hz)"],
                calibration_item[f"Exclusion range {exclusion_idx} (max.) (Hz)"],
            )
            for exclusion_idx in sorted(exclusion_range_indices)
        ],
        "fit_range": (
            calibration_item["Fit range (min.) (Hz)"],
            calibration_item["Fit range (max.) (Hz)"],
        ),
    }

    # If a driving frequency exists, it must have been an active calibration procedure
    calibration_params = read_from_bl_dict(calibration_item, CALIBRATION_PARAM_MAPPING)
    calibration_params["active_calibration"] = bool(
        calibration_item.get("driving_frequency (Hz)", False)
    )

    # If it is not a fixed diode or free diode, it's a fast sensor
    diode_fields = ("Diode frequency (Hz)", "f_diode (Hz)")
    calibration_params["fast_sensor"] = not any(f in calibration_item.keys() for f in diode_fields)

    for key in ("axial", "hydrodynamically_correct"):
        calibration_params[key] = bool(calibration_params[key])

    def check_defined(pair):
        return pair[1] is not None

    return dict(filter(check_defined, (power_spectrum_params | calibration_params).items()))


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
                    items.append(dict(attrs))

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
