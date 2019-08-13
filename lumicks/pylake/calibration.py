from collections import namedtuple


class ForceCalibration:
    """Calibration handling

    Parameters
    ----------
    calibration data : named tuple containing
        time_field - Name of the field used for time
        list of dictionaries containing the raw calibration attribute data
    """
    def __init__(self, calibration):
        self._calibration = calibration

    """filter calibration data based on time stamp range [ns]"""
    @staticmethod
    def _filter_calibration(calibration, start, stop):
        if len(calibration.items) == 0:
            return {}

        def timestamp(x):
            return x[calibration.time_field]

        # Sort by time
        calibration.items = sorted(calibration.items, key=timestamp)
        calibration_items = [calibration.items[0]]
        for i, v in enumerate(calibration.items):
            ts = timestamp(v)
            if ts <= start:
                calibration_items[0] = v
            elif ts < stop:
                calibration_items.append(v)
            else:
                # Since the list is sorted, we can early out
                break

        return calibration_items

    """Filter calibration based on time stamp range

        Parameters
        ----------
        start : time stamp at start [ns]
        stop  : time stamp at stop [ns]
        """
    def filter_calibration(self, start, stop):
        return self.__class__._filter_calibration(self._calibration, start, stop)

    @property
    def calibration(self):
        """Calibration data for this channel"""
        """Calibration data slicing is deferred until calibration is requested to avoid"""
        """slicing values that may be needed."""
        if self._calibration:
            return self.filter_calibration(self.start, self.stop)
        else:
            return AttributeError("No calibration data available")

    def from_dataset(hdf5, n, xy, time_field='Stop time (ns)'):
        """Fetch the force calibration data from the HDF5 file"""
        def parse_force_calibration(cdata, force_idx, force_axis) -> list:
            calibration = namedtuple("Calibration", {"time_field", "items"})
            calibration.time_field = time_field
            calibration.items = []
            for i, v in enumerate(cdata):
                attrs = cdata[v][f"Force {force_idx}{force_axis}"].attrs
                if len(attrs.keys()) > 0 and time_field in attrs.keys():
                    calibration.items.append(dict(attrs))

            return calibration

        if "Calibration" in hdf5.keys():
            if xy:
                calibration_data = parse_force_calibration(hdf5["Calibration"], n, xy)
            else:
                raise NotImplementedError("Calibration is currently only implemented for single axis data")
        else:
            calibration_data = namedtuple("Calibration", {"time_field", "items"})
            calibration_data.time_field = time_field
            calibration_data.items = {}

        return calibration_data
