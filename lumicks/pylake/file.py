import warnings
from typing import Dict

import numpy as np

from .kymo import Kymo
from .note import Note
from .scan import Scan
from .group import Group
from .marker import Marker
from .channel import Slice, TimeTags, Continuous, TimeSeries
from .fdcurve import FdCurve
from .point_scan import PointScan
from .calibration import ForceCalibration
from .detail.mixin import Force, PhotonCounts, DownsampledFD, PhotonTimeTags, BaselineCorrectedForce
from .detail.h5_helper import write_h5

__all__ = ["File"]


class File(Group, Force, DownsampledFD, BaselineCorrectedForce, PhotonCounts, PhotonTimeTags):
    """A convenient HDF5 file wrapper for reading data exported from Bluelake

    Parameters
    ----------
    filename : str | os.PathLike
        The HDF5 file to open in read-only mode

    rgb_to_detectors : Optional[Dict[str, str]]
        Dictionary that maps RGB colors to a photon detector channel (either photon counts, or
        photon time tags). Note that a channel can be left empty by providing the channel name
        "None". Valid colors are ("Red", "Green", "Blue").

    Examples
    --------
    ::

        from lumicks import pylake

        file = pylake.File("example.h5")
        file.force1x.plot()
        file.kymos["name"].plot()

        # Open with custom detector mapping
        file = pylake.File("example.h5", rgb_to_detectors={"Red": "Detector 1", "Green": "Detector 2", "Blue": "Detector 3"})
    """

    SUPPORTED_FILE_FORMAT_VERSIONS = [1, 2]

    def __init__(self, filename, *, rgb_to_detectors=None):
        import h5py

        super().__init__(h5py.File(filename, "r"), lk_file=self)
        self._check_file_format()
        self._rgb_to_detectors = self._get_detector_mapping(rgb_to_detectors)

    def _check_file_format(self):
        if "Bluelake version" not in self.h5.attrs:
            raise Exception("Invalid HDF5 file: no Bluelake version tag found")
        if "File format version" not in self.h5.attrs:
            raise Exception("Invalid HDF5 file: no file format version tag found")
        ff_version = int(self.h5.attrs["File format version"])
        if ff_version not in File.SUPPORTED_FILE_FORMAT_VERSIONS:
            raise Exception(f"Unsupported Bluelake file format version {ff_version}")

        # List of h5 fields for which custom API exists. Note that top level fields (provided as
        # the first argument in the tuple) are automatically printed when print is invoked. For
        # fields where an actual class exists, the second element in the tuple is used to
        # instantiate an element in that Group (using its `from_dataset` method). Otherwise,
        # a FutureWarning is raised.
        self.redirect_list = {
            "Calibration": ("force1x.calibration", None),
            "Marker": ("markers", Marker),
            "FD Curve": ("fdcurves", FdCurve),
            "Kymograph": ("kymos", Kymo),
            "Scan": ("scans", Scan),
            "Note": ("notes", Note),
            "Point Scan": ("point_scans", PointScan),
        }

    def _get_detector_mapping(self, rgb_to_detectors=None):
        """Returns the detector mapping to be used.

        Parameters
        ----------
        rgb_to_detectors : Optional[Dict[str, str]]
            Dictionary that maps RGB colors to a photon detector channel (either photon counts, or
            photon time tags). Note that a channel can be left empty by providing the channel name
            "None".
        """

        def check_custom_detector_mapping():
            for key in rgb_to_detectors.keys():
                if key not in ("Red", "Green", "Blue"):
                    raise ValueError(
                        f'Invalid color mapping ({key}). Valid colors are "Red", "Green" or "Blue"'
                    )

            detectors = set()
            if "Photon time tags" in self.h5:
                detectors = set(self.h5["Photon time tags"])
            elif "Photon count" in self.h5:
                detectors = set(self.h5["Photon count"])

            # Only check if detector data was exported
            if not detectors:
                return

            # "None" indicates that the user doesn't want to plot data for that particular channel.
            if not_found := (set(rgb_to_detectors.values()) - {"None"}) - detectors:
                warnings.warn(
                    RuntimeWarning(
                        f"Invalid RGB to detector mapping: {not_found} photon count channel(s) are "
                        f"missing. Those channels will be blank in images. Available detectors: "
                        f"{detectors}"
                    )
                )

        if rgb_to_detectors:
            check_custom_detector_mapping()
            return rgb_to_detectors
        else:
            return {"Red": "Red", "Green": "Green", "Blue": "Blue"}

    @classmethod
    def from_h5py(cls, h5py_file, *, rgb_to_detectors=None):
        """Directly load an existing `h5py.File <https://docs.h5py.org/en/latest/high/file.html>`_"""
        new_file = cls.__new__(cls)
        new_file.h5 = h5py_file
        new_file._lk_file = new_file
        new_file._check_file_format()
        new_file._rgb_to_detectors = new_file._get_detector_mapping(rgb_to_detectors)
        return new_file

    @property
    def bluelake_version(self) -> str:
        """The version of Bluelake which exported this file"""
        return self.h5.attrs["Bluelake version"]

    @property
    def format_version(self) -> int:
        """The version of the Bluelake-specific HDF5 file structure"""
        return self.h5.attrs["File format version"]

    @property
    def experiment(self) -> str:
        """The name of the experiment as entered by the user in Bluelake"""
        return self.h5.attrs["Experiment"]

    @property
    def description(self) -> str:
        """The description of the measurement as entered by the user in Bluelake"""
        return self.h5.attrs["Description"]

    @property
    def guid(self) -> str:
        """An ID which uniquely identifies each exported file"""
        return self.h5.attrs["GUID"]

    @property
    def export_time(self) -> int:
        """The moment this file was exported"""
        return self.h5.attrs["Export time (ns)"]

    def __repr__(self):
        return f"lumicks.pylake.File('{self.h5.filename}')"

    def __str__(self):
        """Show a quick ASCII overview of the file's contents"""
        import h5py

        def print_attributes(h5file):
            r = "File root metadata:\n"
            for key, value in sorted(h5file.attrs.items()):
                r += f"- {key}: {value}\n"
            return r

        def print_dataset(dset, name, indent):
            space = " " * indent
            r = f"{space}{name}:\n"
            r += f"{space}- Data type: {dset.dtype}\n"
            r += f"{space}- Size: {dset.size}\n"
            return r

        def print_group(group, name="", indent=-2):
            r = ""
            if name:
                more = ":" if len(group) != 0 else ""
                r += f"{' ' * indent}{name}{more}\n"

            for key, item in sorted(group.items()):
                if isinstance(item, h5py.Dataset):
                    r += print_dataset(item, key, indent + 2)
                else:
                    if key not in self.redirect_list:
                        r += print_group(item, key, indent + 2)
            return r

        def print_dicts(field_name):
            field = getattr(self, field_name, None)
            return (
                f"\n.{field_name}\n" + "".join(f"  - {key}\n" for key in field.keys())
                if field
                else ""
            )

        def print_force(field_name):
            field = getattr(self, field_name)
            calibration = "  .calibration\n" if field.calibration else ""
            return f".{field_name}\n{calibration}" if field else ""

        rng = range(4)
        axes = ("x", "y", "z")

        return (
            print_attributes(self.h5)
            + "\n"
            + print_group(self.h5)
            + "".join((print_dicts(field[0]) for field in self.redirect_list.values()))
            + "\n"
            + "".join(
                (
                    print_force(field)
                    for field in [f"force{channel + 1}{axis}" for channel in rng for axis in axes]
                )
            )
            + "\n"
            + "".join(
                (
                    print_force(field)
                    for field in [
                        f"downsampled_force{channel + 1}{axis}" for channel in rng for axis in axes
                    ]
                )
            )
        )

    def _get_force(self, n, xyz):
        """Return a Slice of force measurements, including calibration
        Note: direct access to HDF dataset does not include calibration data"""
        force_group = self.h5["Force HF"][f"Force {n}{xyz}"]
        calibration_data = ForceCalibration.from_dataset(self.h5, n, xyz)

        return Continuous.from_dataset(force_group, "Force (pN)", calibration_data)

    def _get_downsampled_force(self, n, xyz):
        """Return a Slice of low frequency force measurements, including calibration if applicable
        Note: direct access to HDF dataset does not include calibration data"""
        group = self.h5["Force LF"]

        def make(channel):
            if xyz:
                calibration_data = ForceCalibration.from_dataset(self.h5, n, xyz)
                return TimeSeries.from_dataset(group[channel], "Force (pN)", calibration_data)
            else:
                return TimeSeries.from_dataset(group[channel], "Force (pN)")

        if xyz:  # An x, y or z component of the downsampled force is easy
            return make(f"Force {n}{xyz}")

        # Sum force channels can have inconsistent names
        if f"Force {n}" in group:
            return make(f"Force {n}")
        elif f"Trap {n}" in group:
            return make(f"Trap {n}")

        # If it's completely missing, we can reconstruct it from the x and y components, z is not included
        fx = make(f"Force {n}x")
        fy = make(f"Force {n}y")
        return Slice(
            TimeSeries(np.sqrt(fx.data**2 + fy.data**2), fx.timestamps),
            labels={"title": f"Force LF/Force {n}", "y": "Force (pN)"},
        )

    def _get_corrected_force(self, n, xyz):
        """Return a Slice of force measurements, including calibration, with baseline
        correction applied. Only the x-component has correction available.
        Note: direct access to HDF dataset does not include calibration data"""
        force_group = self.h5["Force HF"][f"Corrected Force {n}{xyz}"]
        calibration_data = ForceCalibration.from_dataset(self.h5, n, xyz)

        return Continuous.from_dataset(force_group, "Force (pN)", calibration_data)

    def _get_distance(self, n):
        return TimeSeries.from_dataset(self.h5["Distance"][f"Distance {n}"], r"Distance (μm)")

    def _get_photon_count(self, name):
        return Continuous.from_dataset(
            self.h5["Photon count"][self._rgb_to_detectors[name]], "Photon count"
        )

    def _get_photon_time_tags(self, name):
        return TimeTags.from_dataset(
            self.h5["Photon Time Tags"][self._rgb_to_detectors[name]], "Photon time tags"
        )

    def _get_object_dictionary(self, field, cls):
        def try_from_dataset(*args):
            try:
                return cls.from_dataset(*args)
            except Exception as e:
                warnings.warn(e.args[0])
                return None

        if field not in self.h5:
            return dict()
        scan_objects = [
            (name, try_from_dataset(dset, self)) for name, dset in self.h5[field].items()
        ]
        return {name: scan for name, scan in scan_objects if scan is not None}

    @property
    def kymos(self) -> Dict[str, Kymo]:
        """Kymos stored in the file"""

        # Due to an error in an earlier version of Bluelake, some Kymographs were stored in the
        # `Scan` field. This reads those using a fallback mechanism.
        scan_kymos = {
            key: item
            for key, item in self._get_object_dictionary("Scan", Kymo).items()
            if item._metadata.num_axes == 1
        }

        return scan_kymos | self._get_object_dictionary("Kymograph", Kymo)

    @property
    def point_scans(self) -> Dict[str, Scan]:
        """Point Scans stored in the file"""
        return self._get_object_dictionary("Point Scan", PointScan)

    @property
    def scans(self) -> Dict[str, Scan]:
        """Confocal Scans stored in the file"""
        return self._get_object_dictionary("Scan", Scan)

    @property
    def fdcurves(self) -> Dict[str, FdCurve]:
        """FdCurves stored in the file"""
        return self._get_object_dictionary("FD Curve", FdCurve)

    @property
    def markers(self) -> Dict[str, Marker]:
        """Markers stored in the file"""
        return self._get_object_dictionary("Marker", Marker)

    @property
    def notes(self) -> Dict[str, Note]:
        """Notes stored in the file"""
        return self._get_object_dictionary("Note", Note)

    def save_as(
        self, filename, compression_level=5, omit_data=None, *, crop_time_range=None, verbose=True
    ):
        """Write a modified h5 file to disk.

        When transferring data, it can be beneficial to omit some channels from the h5 file, or use
        a higher compression ratio. High frequency channels tend to take up a lot of space and
        aren't always necessary for every single analysis. It is also worth mentioning that
        Bluelake exports files at compression level 1 for performance reasons, so this function
        can help reduce the file size even when no data is omitted.

        Parameters
        ----------
        filename : str | os.PathLike
            Output file name.
        compression_level : int
            Compression level for gzip compression (default: 5).
        omit_data : str or iterable of str, optional
            Which data sets to omit. Should be a set of h5 paths (e.g. {"Force HF/Force 1y"}).
            `fnmatch` patterns are used to specify which fields to omit, which means you can use
            wildcards as well (see examples below).
        crop_time_range : tuple of np.int64, optional
            Specify a time interval to crop to (tuple of a start and stop time). Interval must be
            specified in nanoseconds since epoch (the same format as timestamps).
        verbose : bool, optional
            Print verbose output. Default: True.

        Examples
        --------
        ::

            import lumicks.pylake as lk

            file = lk.File("example.h5")

            # Saves a file with a high compression level
            file.save_as("smaller.h5", compression_level=9)

            # Omit high frequency force data.
            file.save_as("no_hf.h5", omit_data="Force HF/*")

            # Omit Force 1y data
            file.save_as("no_hf.h5", omit_data="*/Force 1y")

            # Omit Force 1y and 2y data
            file.save_as("no_hf.h5", omit_data=("*/Force 1y", "*/Force 2y"))

            # Omit high frequency force data for channel 1y
            file.save_as("no_1y.h5", omit_data="Force HF/Force 1y")

            # Omit Scan "1"
            file.save_as("no_scan_1.h5", omit_data="Scan/1")

            # Save only the region that contains the kymograph `kymo1`.
            kymo = file.kymos["kymo1"]
            file.save_as("only_kymo.h5", crop_time_range=(kymo.start, kymo.stop))
        """
        write_h5(
            self,
            filename,
            compression_level,
            omit_data,
            crop_time_range=crop_time_range,
            verbose=verbose,
        )
