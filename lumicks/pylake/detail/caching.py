import numpy as np
from cachetools import LRUCache, cached

global_cache = False


def set_cache_enabled(enabled):
    """Enable or disable the global cache

    Pylake offers a global cache. When the global cache is enabled, all `Slice` objects come from
    the same cache.

    Parameters
    ----------
    enabled : bool
        Whether the caching should be enabled (by default it is off)
    """
    global global_cache
    global_cache = enabled


@cached(LRUCache(maxsize=1 << 30, getsizeof=lambda x: x.nbytes), info=True)  # 1 GB of cache
def _get_array(cache_object):
    return cache_object.read_array()


class LazyCache:
    def __init__(self, location, dset):
        """A lazy globally cached wrapper around an object that is convertible to a numpy array"""
        self._location = location
        self._dset = dset

    def __len__(self):
        return len(self._dset)

    def __hash__(self):
        return hash(self._location)

    @staticmethod
    def from_h5py_dset(dset, field=None):
        location = f"{dset.file.filename}{dset.name}"
        if field:
            location = f"{location}.{field}"
            dset = dset.fields(field)

        return LazyCache(location, dset)

    def read_array(self):
        # Note, we deliberately do _not_ allow additional arguments to asarray since we would
        # have to hash those with and unless necessary, they would unnecessarily increase the
        # cache (because of sometimes defensively adding an explicit type). It's better to raise
        # in this case and end up at this comment.
        arr = np.asarray(self._dset)
        arr.flags.writeable = False
        return arr

    def __eq__(self, other):
        return self._location == other._location

    def __array__(self):
        return _get_array(self)


def from_h5py(dset, field=None):
    global global_cache
    return (
        LazyCache.from_h5py_dset(dset, field=field)
        if global_cache
        else dset.fields(field) if field else dset
    )


class LazyCacheMixin:
    def __init__(self):
        self._cache = {}

    def read_lazy_cache(self, key, src_field):
        """A small convenience decorator to incorporate a lazy cache for properties.
        Data will be stored in the `_cache` variable of the instance.

        Parameters
        ----------
        key : str
            Key to use when caching this data
        src_field : LazyCache or dset
            Source field to read from
        """
        global global_cache

        if global_cache:
            return np.asarray(src_field)

        if key not in self._cache:
            self._cache[key] = np.asarray(src_field)

        return self._cache[key]
