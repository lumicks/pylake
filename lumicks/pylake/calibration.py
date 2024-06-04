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
