import warnings

from .channel import channel_class


class Group:
    """A thin wrapper around an HDF5 group with a Bluelake-specific `__getitem__`

    Parameters
    ----------
    h5py_group : h5py.Group
        The underlying h5py group object
    lk_file : lumicks.pylake.File
        Pylake file handle. This is used to construct Pylake objects when they are accessed through
        direct access (e.g. file["Kymograph/kymo"]).

    Attributes
    ----------
    h5 : h5py.Group
        The underlying h5py group object
    """

    def __init__(self, h5py_group, *, lk_file):
        self.h5 = h5py_group
        self._lk_file = lk_file
        self.redirect_list = {}

    def __getitem__(self, item):
        """Return a subgroup or a bluelake timeline channel"""
        import h5py

        thing = self.h5[item]
        item_type = thing.name.split("/")[1]
        redirect_location, redirect_class = self._lk_file.redirect_list.get(item_type, (None, None))
        if redirect_location and not redirect_class:
            warnings.warn(
                f"Direct access to this field is deprecated. Use file.{redirect_location} "
                "instead. In case raw access is needed, go through the fn.h5 directly.",
                FutureWarning,
            )

        if type(thing) is h5py.Group:
            group = Group(thing, lk_file=self._lk_file)
            return group
        else:
            if redirect_location and redirect_class:
                return redirect_class.from_dataset(thing, self._lk_file)
            else:
                cls = channel_class(thing)
                return cls.from_dataset(thing)

    def __iter__(self):
        return self.h5.__iter__()

    def __next__(self):
        return self.h5.__next__()

    def __contains__(self, name):
        return self.h5.__contains__(name)

    def __repr__(self):
        """Return formatted representation of group keys"""
        group_keys = ", ".join(f"'{k}'" for k in self.h5)
        return f"{{{group_keys}}}"
