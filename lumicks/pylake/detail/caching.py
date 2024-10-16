import sys

import numpy as np
from cachetools import LRUCache, keys, cached, cachedmethod

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


_method_cache = LRUCache(maxsize=1 << 30, getsizeof=lambda x: sys.getsizeof(x))  # 1 GB of cache


def method_cache(name):
    """A small convenience decorator to incorporate some really basic instance method memoization

    Note: When used on properties, this one should be included _after_ the @property decorator.
    Data will be stored in the `_cache` variable of the instance.

    Parameters
    ----------
    name : str
        Name of the instance method to memo-ize. Suggestion: the instance method name.

    Examples
    --------
    ::

        class Test:
            def __init__(self):
                self._cache = {}
                ...

            @property
            @method_cache("example_property")
            def example_property(self):
                return 10

            @method_cache("example_method")
            def example_method(self, arguments):
                return 5


        test = Test()
        test.example_property
        test.example_method("hi")
        test._cache
        # test._cache will now show {('example_property',): 10, ('example_method', 'hi'): 5}
    """

    # cachetools>=5.0.0 passes self as first argument. We don't want to bump the reference count
    # by including a reference to the object we're about to store the cache into, so we explicitly
    # drop the first argument. Note that for the default key, they do the same in the package, but
    # we can't use the default key, since it doesn't hash in the method name.
    def key(self, *args, **kwargs):
        return keys.hashkey(self._location, name, *args, **kwargs)

    return cachedmethod(
        lambda self: _method_cache if global_cache and self._location else self._cache, key=key
    )
