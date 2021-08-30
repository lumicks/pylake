import h5py
import warnings
import numpy as np
from typing import Dict

from collections import OrderedDict
from .calibration import ForceCalibration
from .channel import Slice, Continuous, TimeSeries, TimeTags
from .detail.mixin import Force, DownsampledFD, BaselineCorrectedForce, PhotonCounts, PhotonTimeTags
from .detail.h5_helper import write_h5
from .fdcurve import FdCurve
from .group import Group
from .kymo import Kymo
from .point_scan import PointScan
from .scan import Scan
from .marker import Marker

__all__ = ["File"]


class File(Group, Force, DownsampledFD, BaselineCorrectedForce, PhotonCounts, PhotonTimeTags):
    """A convenient HDF5 file wrapper for reading data exported from Bluelake

    Parameters
    ----------
    filename : str
        The HDF5 file to open in read-only mode

    Examples
    --------
    ::

        from lumicks import pylake

        file = pylake.File("example.h5")
        file.force1x.plot()
        file.kymos["name"].plot()
    """

    SUPPORTED_FILE_FORMAT_VERSIONS = [1, 2]

    def __init__(self, filename):
        super().__init__(h5py.File(filename, "r"))
        self._check_file_format()

    def _check_file_format(self):
        if "Bluelake version" not in self.h5.attrs:
            raise Exception("Invalid HDF5 file: no Bluelake version tag found")
        if "File format version" not in self.h5.attrs:
            raise Exception("Invalid HDF5 file: no file format version tag found")
        ff_version = int(self.h5.attrs["File format version"])
        if ff_version not in File.SUPPORTED_FILE_FORMAT_VERSIONS:
            raise Exception(f"Unsupported Bluelake file format version {ff_version}")

        # List of fields which should not be accessed directly from the h5 and their actual API accessors. Note that
        # top level fields are automatically printed when print is invoked.
        self.redirect_list = OrderedDict(
            [
                ("Calibration", "force1x.calibration"),
                ("Marker", "markers"),
                ("FD Curve", "fdcurves"),
                ("Kymograph", "kymos"),
                ("Scan", "scans"),
            ]
        )

    @classmethod
    def from_h5py(cls, h5py_file):
        """Directly load an existing `h5py.File`"""
        new_file = cls.__new__(cls)
        new_file.h5 = h5py_file
        new_file._check_file_format()
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

        return (
            print_attributes(self.h5)
            + "\n"
            + print_group(self.h5)
            + "".join((print_dicts(field) for field in self.redirect_list.values()))
            + "\n"
            + "".join(
                (
                    print_force(field)
                    for field in [
                        "force1x",
                        "force1y",
                        "force1z",
                        "force2x",
                        "force2y",
                        "force2z",
                        "force3x",
                        "force3y",
                        "force3z",
                        "force4x",
                        "force4y",
                        "force4z",
                    ]
                )
            )
            + "\n"
            + "".join(
                (
                    print_force(field)
                    for field in [
                        "downsampled_force1x",
                        "downsampled_force1y",
                        "downsampled_force1z",
                        "downsampled_force2x",
                        "downsampled_force2y",
                        "downsampled_force2z",
                        "downsampled_force3x",
                        "downsampled_force3y",
                        "downsampled_force3z",
                        "downsampled_force4x",
                        "downsampled_force4y",
                        "downsampled_force4z",
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
            TimeSeries(np.sqrt(fx.data ** 2 + fy.data ** 2), fx.timestamps),
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
        return TimeSeries.from_dataset(self.h5["Distance"][f"Distance {n}"], r"Distance ($\mu$m)")

    def _get_photon_count(self, name):
        return Continuous.from_dataset(self.h5["Photon count"][name], "Photon count")

    def _get_photon_time_tags(self, name):
        return TimeTags.from_dataset(self.h5["Photon Time Tags"][name], "Photon time tags")

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
        return self._get_object_dictionary("Kymograph", Kymo)

    @property
    def point_scans(self) -> Dict[str, Scan]:
        return self._get_object_dictionary("Point Scan", PointScan)

    @property
    def scans(self) -> Dict[str, Scan]:
        return self._get_object_dictionary("Scan", Scan)

    @property
    def fdcurves(self) -> Dict[str, FdCurve]:
        return self._get_object_dictionary("FD Curve", FdCurve)

    @property
    def markers(self) -> Dict[str, Marker]:
        return self._get_object_dictionary("Marker", Marker)

    def save_as(self, filename, compression_level=5, omit_data={}):
        """Write a modified h5 file to disk.

        When transferring data, it can be beneficial to omit some channels from the h5 file, or use a higher compression
        ratio. High frequency channels tend to take up a lot of space and aren't always necessary for every single
        analysis. It is also worth mentioning that Bluelake exports files at compression level 1 for performance
        reasons, so this function can help reduce the file size even when no data is omitted.

        Parameters
        ----------
        filename : str
            Output file name.
        compression_level : int
            Compression level for gzip compression (default: 5).
        omit_data : Set[str]
            Which data sets to omit. Should be a set of h5 paths (e.g. {"Force HF/Force 1y"}). `fnmatch` patterns are
            used to specify which fields to omit, which means you can use wildcards as well (see examples below).

        Examples
        --------
        ::

            import lumicks.pylake as lk

            file = lk.File("example.h5")
            file.save_as("smaller.h5", compression_level=9)  # Saves a file with a high compression level

            file.save_as("no_hf.h5", omit_data={"Force HF/*"})  # Omit high frequency force data.

            file.save_as("no_hf.h5", omit_data={"*/Force 1y"})  # Omit Force 1y data

            file.save_as("no_1y.h5", omit_data={"Force HF/Force 1y"})  # Omit high frequency force data for channel 1y
        """
        write_h5(self.h5, filename, compression_level, omit_data)
