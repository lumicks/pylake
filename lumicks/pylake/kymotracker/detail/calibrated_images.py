import matplotlib.pyplot as plt
import numpy as np


class CalibratedKymographChannel:
    """A kymograph image along with its calibration

    Parameters
    ----------
    name : str
        Name of this channel
    data : array_like
        2D image data.
    time_step_ns : float
        Line time [nanoseconds].
    pixel_size : float
        Pixel calibration.
    """

    def __init__(self, name, data, time_step_ns, pixel_size):
        self.name = name
        self.data = data
        self.time_step_ns = time_step_ns
        self._pixel_size = pixel_size

    @classmethod
    def from_array(
        cls,
        image,
        name="from_array",
        time_step_ns=int(1e9),
        pixel_size=1,
    ):
        return cls(
            name,
            image,
            time_step_ns=time_step_ns,
            pixel_size=pixel_size,
        )

    @classmethod
    def from_kymo(cls, kymo, channel):
        return cls(
            kymo.name,
            getattr(kymo, f"{channel}_image"),
            kymo.line_time_seconds * int(1e9),
            kymo.pixelsize_um[0],
        )

    def _to_pixel_rect(self, rect):
        (t0, p0), (t1, p1) = rect
        return [
            [self.from_seconds(t0), self.from_position(p0)],
            [self.from_seconds(t1), self.from_position(p1)],
        ]

    def get_rect(self, rect):
        """Grab a subset of the image in the coordinates of the image

        Parameters
        ----------
        rect : ((float, float), (float, float))
            Rectangle specified in time and positional units according to
            ((min_time, min_position), (max_time, max_position))
        """
        (t0_pixels, p0_pixels), (t1_pixels, p1_pixels) = self._to_pixel_rect(rect)

        if np.any(np.array([t0_pixels, t1_pixels, p0_pixels, p1_pixels]) < 0):
            raise IndexError(
                f"Selection needs to be in bounds of the image (negative coordinates "
                f"are not supported)"
            )

        if p0_pixels > self.data.shape[0]:
            raise IndexError(
                f"Specified minimum position beyond the valid position range {self.to_position(self.data.shape[0])}"
            )

        if t0_pixels > self.data.shape[1]:
            raise IndexError(
                f"Specified minimum time beyond the time range {self.to_seconds(self.data.shape[1])}"
            )

        if t0_pixels >= t1_pixels:
            raise IndexError("Please specify rectangle from minimum time to maximum time")

        if p0_pixels >= p1_pixels:
            raise IndexError("Please specify rectangle from minimum position to maximum position")

        return self.data[p0_pixels:p1_pixels, t0_pixels:t1_pixels]

    @property
    def line_time_seconds(self):
        return self.time_step_ns / 1e9

    def from_position(self, position):
        """Convert from positional coordinates to pixels.

        Note that it rounds the pixel position."""
        return int(position / self._pixel_size)

    def from_seconds(self, time):
        """Convert from seconds to pixels.

        Note that it rounds the pixel position."""
        return int(time / self.line_time_seconds)

    def to_position(self, pixels):
        """Convert from pixels to position"""
        return self._pixel_size * pixels

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
        width_um = self._pixel_size * self.data.shape[0]
        duration = self.line_time_seconds * self.data.shape[1]

        default_kwargs = dict(
            # With origin set to upper (default) bounds should be given as (0, n, n, 0)
            # pixel center aligned with mean time per line
            extent=[
                -0.5 * self.line_time_seconds,
                duration - 0.5 * self.line_time_seconds,
                width_um - 0.5 * self._pixel_size,
                -0.5 * self._pixel_size,
            ],
            aspect=(self.data.shape[0] / self.data.shape[1]) * (duration / width_um),
        )

        plt.imshow(self.data, **{**default_kwargs, **kwargs})
        plt.xlabel("time (s)")
        plt.ylabel(r"position ($\mu$m)")
        plt.title(self.name)
