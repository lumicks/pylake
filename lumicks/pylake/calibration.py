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
    def from_dataset(hdf5, n, xy, time_field="Stop time (ns)"):
        """Fetch the force calibration data from the HDF5 file"""

        def parse_force_calibration(cdata, force_idx, force_axis) -> list:
            items = []
            for v in cdata:
                attrs = cdata[v][f"Force {force_idx}{force_axis}"].attrs
                if time_field in attrs.keys():
                    items.append(dict(attrs))

            calibration = ForceCalibration(time_field=time_field, items=items)

            return calibration

        if "Calibration" in hdf5.keys():
            if xy:
                calibration_data = parse_force_calibration(hdf5["Calibration"], n, xy)
            else:
                raise NotImplementedError(
                    "Calibration is currently only implemented for single axis data"
                )
        else:
            calibration_data = ForceCalibration(time_field=time_field, items=[])

        return calibration_data
