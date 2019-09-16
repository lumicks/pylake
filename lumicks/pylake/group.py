import h5py
from .channel import channel_class


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
            cls = channel_class(thing)
            return cls.from_dataset(thing)

    def __iter__(self):
        return self.h5.__iter__()

    def __next__(self):
        return self.h5.__next__()

    def __repr__(self):
        """Return formatted representation of group keys"""
        group_keys = ", ".join(f"'{k}'" for k in self.h5)
        return f"{{{group_keys}}}"
