class ForceCalibration:
    """Parameters
    ----------
    calibration data : list of dictionaries
        Calibration data with a field called "Stop time (ns)"
    """
    def __init__(self, calibration):
        self._calibration = calibration

    """Calibration data"""
    """Filter calibration data based on time stamp range [ns]"""
    @staticmethod
    def _filter_calibration(calibration, start, stop):
        def timestamp(x):
            return x['Stop time (ns)']

        # Sort by time
        calibration = sorted(calibration, key=timestamp)
        calibration_items = [calibration[0]]
        for i, v in enumerate(calibration):
            ts = timestamp(v)
            if ts <= start:
                calibration_items[0] = v
            elif ts < stop:
                calibration_items.append(v)
            else:
                # Since the list is sorted, we can early out
                break

        return calibration_items

    def calibration(self):
        """Calibration data for this channel"""
        """Calibration data slicing is deferred until calibration is requested to avoid"""
        """slicing values that may be needed."""
        if self._calibration:
            return self.__class__._filter_calibration(self._calibration, self.start, self.stop)
        else:
            return AttributeError("No calibration data available")

    def from_dataset(hdf5, n, xy):
        """Fetch the force calibration data from the HDF5 file"""
        def parse_force_calibration(cdata, force_idx, force_axis) -> list:
            calibration_src = []
            for i, v in enumerate(cdata):
                calibration_src.append(dict(cdata[v][f"Force {force_idx}{force_axis}"].attrs))

            return calibration_src

        if "Calibration" in hdf5.keys():
            calibration_data = parse_force_calibration(hdf5["Calibration"], n, xy)
        else:
            calibration_data = {}

        return calibration_data
