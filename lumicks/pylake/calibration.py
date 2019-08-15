from collections import namedtuple

"""A source of calibration data

Parameters
----------
time_field : string
    name of the field used for time
items : list
    list of dictionaries containing raw calibration attribute data
"""
Calibration = namedtuple("Calibration", {"time_field", "items"})


def _filter_calibration(calibration, start, stop):
    """filter calibration data based on time stamp range [ns]"""
    if len(calibration.items) == 0:
        return []

    def timestamp(x):
        return x[calibration.time_field]

    calibration.items = sorted(calibration.items, key=timestamp)

    calibration_items = [x for x in calibration.items if start <= timestamp(x) < stop]
    pre = [x for x in calibration.items if timestamp(x) <= start]
    if pre and (not calibration_items or start < timestamp(calibration_items[0])):
        calibration_items.insert(0, pre[-1])

    return calibration_items


class ForceCalibration:
    """Calibration handling

    Parameters
    ----------
    calibration: Calibration
        Calibration information
    """
    def __init__(self, calibration):
        if not calibration:
            raise ValueError("Missing argument calibration")

        self._src = calibration

    def filter_calibration(self, start, stop):
        """Filter calibration based on time stamp range

            Parameters
            ----------
            start : int
                time stamp at start [ns]
            stop  : int
                time stamp at stop [ns]"""
        return _filter_calibration(self._src, start, stop)

    @property
    def calibration(self):
        """Calibration data for this channel

        Calibration data slicing is deferred until calibration is requested to avoid
        slicing values that may be needed.
        """
        return self.filter_calibration(self.start, self.stop)

    @staticmethod
    def from_dataset(hdf5, n, xy, time_field='Stop time (ns)'):
        """Fetch the force calibration data from the HDF5 file"""
        def parse_force_calibration(cdata, force_idx, force_axis) -> list:
            calibration = Calibration
            calibration.time_field = time_field
            calibration.items = []
            for v in cdata:
                attrs = cdata[v][f"Force {force_idx}{force_axis}"].attrs
                if time_field in attrs.keys():
                    calibration.items.append(dict(attrs))

            return calibration

        if "Calibration" in hdf5.keys():
            if xy:
                calibration_data = parse_force_calibration(hdf5["Calibration"], n, xy)
            else:
                raise NotImplementedError("Calibration is currently only implemented for single axis data")
        else:
            calibration_data = Calibration
            calibration_data.time_field = time_field
            calibration_data.items = []

        return calibration_data
