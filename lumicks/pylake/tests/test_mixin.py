from lumicks.pylake.detail import mixin
import itertools


def test_force():
    properties = itertools.product([1, 2, 3, 4], ["x", "y"])

    # Ensure that all properties map to the correct channels
    class Mapping(mixin.Force):
        def _get_force(self, n, xy):
            return f"{n}{xy}"

    a = Mapping()
    for n, xy in properties:
        assert getattr(a, f"force{n}{xy}") == f"{n}{xy}"

    class Empty(mixin.Force):
        def _get_force(self, n, xy):
            raise KeyError

    # Ensure that missing channels return empty objects
    b = Empty()
    for n, xy in properties:
        assert not getattr(b, f"force{n}{xy}")


def test_downsampled_fd():
    force_properties = itertools.product([1, 2, 3, 4], ["x", "y"])
    distance_properties = [1, 2]

    class Mapping(mixin.DownsampledFD):
        def _get_downsampled_force(self, n, xy):
            return f"{n}{xy}"

        def _get_distance(self, n):
            return f"{n}"

    a = Mapping()
    for n, xy in force_properties:
        assert getattr(a, f"downsampled_force{n}{xy}") == f"{n}{xy}"

    for n in distance_properties:
        assert getattr(a, f"distance{n}") == f"{n}"

    class Empty(mixin.DownsampledFD):
        def _get_downsampled_force(self, n, xy):
            raise KeyError

        def _get_distance(self, n):
            raise KeyError

    b = Empty()
    for n, xy in force_properties:
        assert not getattr(b, f"downsampled_force{n}{xy}")

    for n in distance_properties:
        assert not getattr(b, f"distance{n}")


def test_photon_count():
    properties = ["red", "green", "blue"]

    class Mapping(mixin.PhotonCounts):
        def _get_photon_count(self, name):
            return f"{name}".lower()

    a = Mapping()
    for name in properties:
        assert getattr(a, f"{name}_photon_count") == f"{name}"

    class Empty(mixin.PhotonCounts):
        def _get_photon_count(self, name):
            raise KeyError

    b = Empty()
    for name in properties:
        assert not getattr(b, f"{name}_photon_count")
