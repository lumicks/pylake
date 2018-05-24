import h5py
from .channel import is_continuous_channel, make_continuous_channel, make_timeseries_channel


class Group:
    """A thin wrapper around an HDF5 group with a Bluelake-specific `__getitem__`

    Attributes
    ----------
    h5 : h5py.Group
        The underlying h5py group object
    """
    def __init__(self, h5py_group):
        self.h5 = h5py_group

    def __getitem__(self, item):
        """Return a subgroup or a bluelake timeline channel"""
        thing = self.h5[item]
        if type(thing) is h5py.Group:
            return Group(thing)
        else:
            if is_continuous_channel(thing):
                return make_continuous_channel(thing)
            else:
                return make_timeseries_channel(thing)
