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
        self._idx = -1

    def __getitem__(self, item):
        """Return a subgroup or a bluelake timeline channel"""
        thing = self.h5[item]
        if type(thing) is h5py.Group:
            return Group(thing)
        else:
            cls = channel_class(thing)
            return cls.from_dataset(thing)

    def keys(self):
        """Return group names at this level"""
        return list(self.h5.keys())

    def __iter__(self):
        self._idx = -1
        return self

    def __next__(self):
        """Return key names for the iterator analogously to HDF5"""
        if self._idx >= len(self.keys()) - 1:
            raise StopIteration

        self._idx += 1
        return self.keys()[self._idx]

    def __repr__(self):
        """Return type name and members of the group"""
        name = self.__class__.__name__
        members = ''.join(f"{e}, " for e in self.keys())
        return f"{name} (members: {members[:-2]})"
