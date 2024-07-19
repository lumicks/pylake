import datetime

from tabulate import tabulate

from lumicks.pylake.force_calibration.calibration_item import ForceCalibrationItem


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


class ForceCalibrationList:
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
        return ForceCalibrationList(self._time_field, _src)

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
        return ForceCalibrationList(
            self._time_field,
            _filter_calibration(self._time_field, self._src, start, stop),
            start,
            stop,
        )

    @staticmethod
    def from_field(hdf5, force_channel, time_field="Stop time (ns)") -> "ForceCalibrationList":
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
            return ForceCalibrationList(time_field=time_field, items=[])

        items = []
        for calibration_item in hdf5["Calibration"].values():
            if force_channel in calibration_item:
                attrs = dict(calibration_item[force_channel].attrs)
                if time_field in attrs.keys():
                    # Copy the timestamp at which the calibration was applied into the item
                    attrs["Timestamp (ns)"] = calibration_item.attrs.get("Timestamp (ns)")
                    items.append(ForceCalibrationItem(attrs))

        return ForceCalibrationList(time_field=time_field, items=items)

    def _print_summary(self, tablefmt):
        def format_timestamp(timestamp):
            return datetime.datetime.fromtimestamp(int(timestamp)).strftime("%x %X")

        return tabulate(
            (
                (
                    idx,
                    format_timestamp(item.applied_at / 1e9) if item.applied_at else "-",
                    item.kind,
                    f"{item.stiffness:.2f}" if item.stiffness else "N/A",
                    f"{item.force_sensitivity:.2f}" if item.force_sensitivity else "N/A",
                    (
                        f"{item.displacement_sensitivity:.2f}"
                        if item.displacement_sensitivity
                        else "N/A"
                    ),
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
                "#",
                "Applied at",
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
        return self._print_summary(tablefmt="html")

    def __str__(self):
        return self._print_summary(tablefmt="text")

    @staticmethod
    def from_dataset(hdf5, n, xy, time_field="Stop time (ns)") -> "ForceCalibrationList":
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
            return ForceCalibrationList.from_field(
                hdf5, force_channel=f"Force {n}{xy}", time_field=time_field
            )
        else:
            raise NotImplementedError(
                "Calibration is currently only implemented for single axis data"
            )
