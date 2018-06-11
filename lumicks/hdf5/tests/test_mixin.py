from lumicks.hdf5.detail import mixin
import itertools


def test_force():
    class A(mixin.Force):
        def _get_force(self, n, xy):
            return f"{n}{xy}"

    a = A()
    for n, xy in itertools.product([1, 2, 3, 4], ["x", "y"]):
        assert getattr(a, f"force{n}{xy}") == f"{n}{xy}"


def test_downsampled_fd():
    class A(mixin.DownsampledFD):
        def _get_downsampled_force(self, n, xy):
            return f"{n}{xy}"

        def _get_distance(self, n):
            return f"{n}"

    a = A()
    for n, xy in itertools.product([1, 2, 3, 4], ["x", "y"]):
        assert getattr(a, f"downsampled_force{n}{xy}") == f"{n}{xy}"

    for n in [1, 2]:
        assert getattr(a, f"distance{n}") == f"{n}"


def test_photon_count():
    class A(mixin.PhotonCounts):
        def _get_photon_count(self, name):
            return f"{name}".lower()

    a = A()
    for name in ["red", "green", "blue"]:
        assert getattr(a, f"{name}_photons") == f"{name}"
