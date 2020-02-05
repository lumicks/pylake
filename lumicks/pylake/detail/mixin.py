"""Mixin class which add properties for predefined channels"""
from ..channel import Slice, empty_slice


def _try_get_or_empty(f, *args, **kwargs):
    """Try to call `f` or return an empty slice on `KeyError`"""
    try:
        return f(*args, **kwargs)
    except KeyError:
        return empty_slice


class Force:
    """Full frequency force channels"""
    def _get_force(self, n, xy):
        raise NotImplementedError

    @property
    def force1x(self) -> Slice:
        return _try_get_or_empty(self._get_force, 1, "x")

    @property
    def force1y(self) -> Slice:
        return _try_get_or_empty(self._get_force, 1, "y")

    @property
    def force2x(self) -> Slice:
        return _try_get_or_empty(self._get_force, 2, "x")

    @property
    def force2y(self) -> Slice:
        return _try_get_or_empty(self._get_force, 2, "y")

    @property
    def force3x(self) -> Slice:
        return _try_get_or_empty(self._get_force, 3, "x")

    @property
    def force3y(self) -> Slice:
        return _try_get_or_empty(self._get_force, 3, "y")

    @property
    def force4x(self) -> Slice:
        return _try_get_or_empty(self._get_force, 4, "x")

    @property
    def force4y(self) -> Slice:
        return _try_get_or_empty(self._get_force, 4, "y")


class DownsampledFD:
    """Downsampled force and distance channels"""
    def _get_downsampled_force(self, n, xy):
        raise NotImplementedError

    def _get_distance(self, n):
        raise NotImplementedError

    @property
    def downsampled_force1(self) -> Slice:
        return _try_get_or_empty(self._get_downsampled_force, 1, "")

    @property
    def downsampled_force2(self) -> Slice:
        return _try_get_or_empty(self._get_downsampled_force, 2, "")

    @property
    def downsampled_force3(self) -> Slice:
        return _try_get_or_empty(self._get_downsampled_force, 3, "")

    @property
    def downsampled_force4(self) -> Slice:
        return _try_get_or_empty(self._get_downsampled_force, 4, "")

    @property
    def downsampled_force1x(self) -> Slice:
        return _try_get_or_empty(self._get_downsampled_force, 1, "x")

    @property
    def downsampled_force1y(self) -> Slice:
        return _try_get_or_empty(self._get_downsampled_force, 1, "y")

    @property
    def downsampled_force2x(self) -> Slice:
        return _try_get_or_empty(self._get_downsampled_force, 2, "x")

    @property
    def downsampled_force2y(self) -> Slice:
        return _try_get_or_empty(self._get_downsampled_force, 2, "y")

    @property
    def downsampled_force3x(self) -> Slice:
        return _try_get_or_empty(self._get_downsampled_force, 3, "x")

    @property
    def downsampled_force3y(self) -> Slice:
        return _try_get_or_empty(self._get_downsampled_force, 3, "y")

    @property
    def downsampled_force4x(self) -> Slice:
        return _try_get_or_empty(self._get_downsampled_force, 4, "x")

    @property
    def downsampled_force4y(self) -> Slice:
        return _try_get_or_empty(self._get_downsampled_force, 4, "y")

    @property
    def distance1(self) -> Slice:
        return _try_get_or_empty(self._get_distance, 1)

    @property
    def distance2(self) -> Slice:
        return _try_get_or_empty(self._get_distance, 2)


class PhotonCounts:
    """Red, green and blue photon channels"""
    def _get_photon_count(self, name):
        raise NotImplementedError

    @property
    def red_photon_count(self) -> Slice:
        return _try_get_or_empty(self._get_photon_count, "Red")

    @property
    def green_photon_count(self) -> Slice:
        return _try_get_or_empty(self._get_photon_count, "Green")

    @property
    def blue_photon_count(self) -> Slice:
        return _try_get_or_empty(self._get_photon_count, "Blue")


class PhotonTimeTags:
    """Red, green, and blue photon time tag channels"""
    def _get_photon_time_tags(self, name):
        raise NotImplementedError

    @property
    def red_photon_time_tags(self) -> Slice:
        return _try_get_or_empty(self._get_photon_time_tags, "Red")

    @property
    def green_photon_time_tags(self) -> Slice:
        return _try_get_or_empty(self._get_photon_time_tags, "Green")

    @property
    def blue_photon_time_tags(self) -> Slice:
        return _try_get_or_empty(self._get_photon_time_tags, "Blue")

class ExcitationLaserPower:
    """Red, green, blue, and sted laser power"""
    def _get_laser_power(self, name):
        power_data = self.file["Confocal diagnostics"][f"Excitation Laser {name}"]
        # fetch the timestamp of the last datapoint before the beginning of the item
        start_time = power_data[:self.start].timestamps[-1]
        return power_data[start_time:self.stop]
        
    @property
    def red_power(self) -> Slice:
        return _try_get_or_empty(self._get_laser_power, "Red")

    @property
    def green_power(self) -> Slice:
        return _try_get_or_empty(self._get_laser_power, "Green")

    @property
    def blue_power(self) -> Slice:
        return _try_get_or_empty(self._get_laser_power, "Blue")

    @property
    def sted_power(self) -> Slice:
        return _try_get_or_empty(self._get_laser_power, "Sted")
