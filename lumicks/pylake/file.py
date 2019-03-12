import h5py
import numpy as np
from typing import Dict

from .channel import Slice, Continuous, TimeSeries, TimeTags, channel_class
from .detail.mixin import Force, DownsampledFD, PhotonCounts, PhotonTimeTags
from .fdcurve import FDCurve
from .group import Group
from .kymo import Kymo
from .point_scan import PointScan
from .scan import Scan

__all__ = ["File"]


class File(Group, Force, DownsampledFD, PhotonCounts, PhotonTimeTags):
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
        super().__init__(h5py.File(filename, 'r'))
        self._check_file_format()

    def _check_file_format(self):
        if "Bluelake version" not in self.h5.attrs:
            raise Exception("Invalid HDF5 file: no Bluelake version tag found")
        if "File format version" not in self.h5.attrs:
            raise Exception("Invalid HDF5 file: no file format version tag found")
        ff_version = int(self.h5.attrs["File format version"])
        if ff_version not in File.SUPPORTED_FILE_FORMAT_VERSIONS:
            raise Exception(f"Unsupported Bluelake file format version {ff_version}")

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
                    r += print_group(item, key, indent + 2)
            return r

        return print_attributes(self.h5) + "\n" + print_group(self.h5)

    def _get_force(self, n, xy):
        return Continuous.from_dataset(self.h5["Force HF"][f"Force {n}{xy}"], "Force (pN)")

    def _get_downsampled_force(self, n, xy):
        group = self.h5["Force LF"]

        def make(channel):
            return TimeSeries.from_dataset(group[channel], "Force (pN)")

        if xy:  # An x or y component of the downsampled force is easy
            return make(f"Force {n}{xy}")

        # Sum force channels can have inconsistent names
        if f"Force {n}" in group:
            return make(f"Force {n}")
        elif f"Trap {n}" in group:
            return make(f"Trap {n}")

        # If it's completely missing, we can reconstruct it from the x and y components
        fx = make(f"Force {n}x")
        fy = make(f"Force {n}y")
        return Slice(TimeSeries(np.sqrt(fx.data**2 + fy.data**2), fx.timestamps),
                     labels={"title": f"Force LF/Force {n}", "y": "Force (pN)"})

    def _get_distance(self, n):
        return TimeSeries.from_dataset(self.h5["Distance"][f"Distance {n}"],
                                       r"Distance ($\mu$m)")

    def _get_photon_count(self, name):
        return Continuous.from_dataset(self.h5["Photon count"][name], "Photon count")

    def _get_photon_time_tags(self, name):
        return TimeTags.from_dataset(self.h5["Photon Time Tags"][name], "Photon time tags")

    @property
    def kymos(self) -> Dict[str, Kymo]:
        if "Kymograph" not in self.h5:
            return dict()
        return {name: Kymo(dset, self) for name, dset in self.h5["Kymograph"].items()}

    @property
    def point_scans(self) -> Dict[str, Scan]:
        if "Point Scan" not in self.h5:
            return dict()
        return {name: PointScan(dset, self) for name, dset in self.h5["Point Scan"].items()}

    @property
    def scans(self) -> Dict[str, Scan]:
        if "Scan" not in self.h5:
            return dict()
        return {name: Scan(dset, self) for name, dset in self.h5["Scan"].items()}

    @property
    def fdcurves(self) -> Dict[str, FDCurve]:
        if "FD Curve" not in self.h5:
            return dict()
        return {name: FDCurve.from_dset(dset, self) for name, dset in self.h5["FD Curve"].items()}
