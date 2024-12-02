import pytest

from lumicks.pylake.detail import caching


@pytest.mark.parametrize(
    "location, use_global_cache",
    [
        (None, False),
        (None, True),
        ("test", False),
        ("test", True),
    ],
)
def test_cache_method(location, use_global_cache):
    calls = 0

    def call():
        nonlocal calls
        calls += 1

    class Test:
        def __init__(self, location):
            self._cache = {}
            self._location = location

        @property
        @caching.method_cache("example_property")
        def example_property(self):
            call()
            return 10

        @caching.method_cache("example_method")
        def example_method(self, argument=5):
            call()
            return argument

    old_cache = caching.global_cache
    caching.set_cache_enabled(use_global_cache)
    caching._method_cache.clear()
    test = Test(location=location)

    cache_location = caching._method_cache if use_global_cache and location else test._cache

    assert len(cache_location) == 0
    assert test.example_property == 10
    assert len(cache_location) == 1
    assert calls == 1
    assert test.example_property == 10
    assert calls == 1
    assert len(cache_location) == 1

    assert test.example_method() == 5
    assert calls == 2
    assert len(cache_location) == 2

    assert test.example_method() == 5
    assert calls == 2
    assert len(cache_location) == 2

    assert test.example_method(6) == 6
    assert calls == 3
    assert len(cache_location) == 3

    assert test.example_method(6) == 6
    assert calls == 3
    assert len(cache_location) == 3

    assert test.example_method() == 5
    assert calls == 3
    assert len(cache_location) == 3

    caching.set_cache_enabled(old_cache)
