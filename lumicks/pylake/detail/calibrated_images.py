import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce


class CalibratedKymographChannel:
    """A kymograph image along with its calibration"""

    def __init__(self, name, data, start, time_step, calibration, downsampling_factor=1):
        self.name = name
        self.data = data
        self.start = start
        self.time_step = time_step
        self._calibration = calibration
        self.downsampling_factor = downsampling_factor

    @classmethod
    def from_array(cls, image, start=0, time_step=int(1e9), calibration=1, downsampling_factor=1):
        return cls(
            "test",
            image,
            start=start,
            time_step=time_step,
            calibration=calibration,
            downsampling_factor=downsampling_factor,
        )

    @classmethod
    def from_kymo(cls, kymo, channel):
        ts = kymo.timestamps[0, :]
        time_steps = np.unique(np.diff(ts))
        if len(time_steps) > 1:
            raise RuntimeError("Each line should have the same scan time.")

        return cls(
            kymo.name, getattr(kymo, f"{channel}_image"), ts[0], time_steps[0], kymo.pixelsize_um[0]
        )

    def _get_rect(self, rect):
        """Grab a subset of the image in the coordinates of the image"""
        ((t0, p0), (t1, p1)) = rect
        t0_pixels, t1_pixels = self.from_seconds(t0), self.from_seconds(t1)
        p0_pixels, p1_pixels = self.from_coord(p0), self.from_coord(p1)

        if p0_pixels > self.data.shape[0]:
            raise IndexError(
                f"Specified minimum position {p0} beyond the valid coordinate range {self.to_coord(self.data.shape[0])}"
            )

        if t0_pixels > self.data.shape[1]:
            raise IndexError(
                f"Specified minimum time {t0} beyond the time range {self.to_seconds(self.data.shape[1])}"
            )

        if t0 >= t1:
            raise IndexError("Please specify rectangle from minimum time to maximum time")

        if p0 >= p1:
            raise IndexError("Please specify rectangle from minimum position to maximum position")

        return self.data[p0_pixels:p1_pixels, t0_pixels:t1_pixels]

    def downsampled_by(self, factor, reduce=np.sum):
        """Downsample the time axis by factor. If the last block is not filled, it is omitted from
        the result.

        Parameters
        ----------
        factor : int
            Factor by which to downsample the time axis.
        reduce : callable
            Function applied to each block when downsampling (Default: np.sum).
        """
        return CalibratedKymographChannel(
            self.name,
            block_reduce(self.data, (1, factor), func=reduce)[:, : self.data.shape[1] // factor],
            self.start,
            self.time_step * factor,
            self._calibration,
            self.downsampling_factor * factor,
        )

    @property
    def line_time_seconds(self):
        return self.time_step / 1e9

    def from_coord(self, coord):
        return int(coord / self._calibration)

    def from_seconds(self, time):
        return int(time / self.line_time_seconds)

    def to_coord(self, pixels):
        """Convert from pixels to the coordinate system of this image"""
        return self._calibration * pixels

    def to_seconds(self, x):
        """Convert from pixels to time in seconds"""
        return self.line_time_seconds * x

    def plot(self, **kwargs):
        """Plot the calibrated image

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        # TODO: Deduplicate this with Kymo after tests are refactored to incorporate a constant
        #  linetime kymograph.
        width_um = self._calibration * self.data.shape[0]
        duration = self.line_time_seconds * self.data.shape[1]

        default_kwargs = dict(
            # With origin set to upper (default) bounds should be given as (0, n, n, 0)
            # pixel center aligned with mean time per line
            extent=[
                -0.5 * self.line_time_seconds,
                duration - 0.5 * self.line_time_seconds,
                width_um - 0.5 * self._calibration,
                -0.5 * self._calibration,
            ],
            aspect=(self.data.shape[0] / self.data.shape[1]) * (duration / width_um),
        )

        plt.imshow(self.data, **{**default_kwargs, **kwargs})
        plt.xlabel("time (s)")
        plt.ylabel(r"position ($\mu$m)")
        plt.title(self.name)
