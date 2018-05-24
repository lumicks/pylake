import h5py
from .group import Group
from .channel import make_continuous_channel, make_timeseries_channel


class File(Group):
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

    def _get_force(self, n, xy):
        return make_continuous_channel(self.h5["Force HF"][f"Force {n}{xy}"], "Force (pN)")

    @property
    def force1x(self):
        return self._get_force(1, "x")

    @property
    def force1y(self):
        return self._get_force(1, "y")

    @property
    def force2x(self):
        return self._get_force(2, "x")

    @property
    def force2y(self):
        return self._get_force(2, "y")

    @property
    def force3x(self):
        return self._get_force(3, "x")

    @property
    def force3y(self):
        return self._get_force(3, "y")

    @property
    def force4x(self):
        return self._get_force(4, "x")

    @property
    def force4y(self):
        return self._get_force(4, "y")

    def _get_downsampled_force(self, n, xy):
        return make_timeseries_channel(self.h5["Force LF"][f"Force {n}{xy}"], "Force (pN)")

    @property
    def downsampled_force1x(self):
        return self._get_downsampled_force(1, "x")

    @property
    def downsampled_force1y(self):
        return self._get_downsampled_force(1, "y")

    @property
    def downsampled_force2x(self):
        return self._get_downsampled_force(2, "x")

    @property
    def downsampled_force2y(self):
        return self._get_downsampled_force(2, "y")

    @property
    def downsampled_force3x(self):
        return self._get_downsampled_force(3, "x")

    @property
    def downsampled_force3y(self):
        return self._get_downsampled_force(3, "y")

    @property
    def downsampled_force4x(self):
        return self._get_downsampled_force(4, "x")

    @property
    def downsampled_force4y(self):
        return self._get_downsampled_force(4, "y")

    def _get_distance(self, n):
        return make_timeseries_channel(self.h5["Distance"][f"Distance {n}"],
                                       r"Distance ($\mu$m)")

    @property
    def distance1(self):
        return self._get_distance(1)

    @property
    def distance2(self):
        return self._get_distance(2)

    def _get_photon_count(self, name):
        return make_continuous_channel(self.h5["Photon count"][name], "Photon count")

    @property
    def red_photons(self):
        return self._get_photon_count("Red")

    @property
    def green_photons(self):
        return self._get_photon_count("Green")

    @property
    def blue_photons(self):
        return self._get_photon_count("Blue")
