"""Mixin class which add properties for predefined channels"""
from ..channel import Slice


class Force:
    """Full frequency force channels"""
    def _get_force(self, n, xy):
        raise NotImplementedError

    @property
    def force1x(self) -> Slice:
        return self._get_force(1, "x")

    @property
    def force1y(self) -> Slice:
        return self._get_force(1, "y")

    @property
    def force2x(self) -> Slice:
        return self._get_force(2, "x")

    @property
    def force2y(self) -> Slice:
        return self._get_force(2, "y")

    @property
    def force3x(self) -> Slice:
        return self._get_force(3, "x")

    @property
    def force3y(self) -> Slice:
        return self._get_force(3, "y")

    @property
    def force4x(self) -> Slice:
        return self._get_force(4, "x")

    @property
    def force4y(self) -> Slice:
        return self._get_force(4, "y")


class DownsampledFD:
    """Downsampled force and distance channels"""
    def _get_downsampled_force(self, n, xy):
        raise NotImplementedError

    def _get_distance(self, n):
        raise NotImplementedError

    @property
    def downsampled_force1x(self) -> Slice:
        return self._get_downsampled_force(1, "x")

    @property
    def downsampled_force1y(self) -> Slice:
        return self._get_downsampled_force(1, "y")

    @property
    def downsampled_force2x(self) -> Slice:
        return self._get_downsampled_force(2, "x")

    @property
    def downsampled_force2y(self) -> Slice:
        return self._get_downsampled_force(2, "y")

    @property
    def downsampled_force3x(self) -> Slice:
        return self._get_downsampled_force(3, "x")

    @property
    def downsampled_force3y(self) -> Slice:
        return self._get_downsampled_force(3, "y")

    @property
    def downsampled_force4x(self) -> Slice:
        return self._get_downsampled_force(4, "x")

    @property
    def downsampled_force4y(self) -> Slice:
        return self._get_downsampled_force(4, "y")

    @property
    def distance1(self) -> Slice:
        return self._get_distance(1)

    @property
    def distance2(self) -> Slice:
        return self._get_distance(2)


class PhotonCounts:
    """Red, green and blue photon channels"""
    def _get_photon_count(self, name):
        raise NotImplementedError

    @property
    def red_photons(self) -> Slice:
        return self._get_photon_count("Red")

    @property
    def green_photons(self) -> Slice:
        return self._get_photon_count("Green")

    @property
    def blue_photons(self) -> Slice:
        return self._get_photon_count("Blue")
