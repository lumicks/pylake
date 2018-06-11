import h5py
from typing import Dict

from .channel import make_continuous_channel, make_timeseries_channel
from .detail.mixin import Force, DownsampledFD, PhotonCounts
from .fdcurve import FDCurve
from .group import Group
from .kymo import Kymo


class File(Group, Force, DownsampledFD, PhotonCounts):
    """A convenient HDF5 file wrapper for reading data exported from Bluelake

    Parameters
    ----------
    filename : str
        The HDF5 file to open in read-only mode

    Examples
    --------
    ```
    file = lumicks.hdf5.File("example.h5")
    file.force1x.plot()
    file.kymos["name"].plot()
    ```
    """
    def __init__(self, filename):
        super().__init__(h5py.File(filename, 'r'))

    @classmethod
    def from_h5py(cls, h5py_file):
        """Directly load an existing `h5py.File`"""
        new_file = cls.__new__(cls)
        super(cls, new_file).__init__(h5py_file)
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
        return f"lumicks.hdf5.File('{self.h5.filename}')"

    def __str__(self):
        """Show a quick ASCII overview of the file's contents"""
        def print_attributes(file):
            r = "File root metadata:\n"
            for key, value in self.h5.attrs.items():
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

            for key, item in group.items():
                if isinstance(item, h5py.Dataset):
                    r += print_dataset(item, key, indent + 2)
                else:
                    r += print_group(item, key, indent + 2)
            return r

        return print_attributes(self.h5) + "\n" + print_group(self.h5)

    def _get_force(self, n, xy):
        return make_continuous_channel(self.h5["Force HF"][f"Force {n}{xy}"], "Force (pN)")

    def _get_downsampled_force(self, n, xy):
        return make_timeseries_channel(self.h5["Force LF"][f"Force {n}{xy}"], "Force (pN)")

    def _get_distance(self, n):
        return make_timeseries_channel(self.h5["Distance"][f"Distance {n}"],
                                       r"Distance ($\mu$m)")

    def _get_photon_count(self, name):
        return make_continuous_channel(self.h5["Photon count"][name], "Photon count")

    @property
    def kymos(self) -> Dict[str, Kymo]:
        return {name: Kymo(dset, self) for name, dset in self.h5["Kymograph"].items()}

    @property
    def fdcurves(self) -> Dict[str, FDCurve]:
        return {name: FDCurve(dset, self) for name, dset in self.h5["FD Curve"].items()}
