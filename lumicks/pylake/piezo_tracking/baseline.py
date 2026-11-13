import numpy as np

from lumicks.pylake.channel import Slice, Continuous, _match_sample_rates

__all__ = ["ForceBaseLine"]


def _match_piezo_channels(channels):
    """Crop native channels to complete shared samples and match their sample rates."""
    message = "No complete overlapping samples in the provided channels."
    if not channels or any(len(channel) == 0 for channel in channels):
        raise ValueError(message)

    start = max(channel.start for channel in channels)
    stop = min(channel.stop for channel in channels)
    channels = [channel[start:stop] for channel in channels]
    if any(len(channel) == 0 for channel in channels):
        raise ValueError(message)

    # Slicing can advance the slower channel to its next sample.
    #
    # Consider the following example:
    # Position:  0          50          100          150
    # Force:        10 20 30 40 50 60 … 100 …
    #
    # First start = max(...) gives 10
    # Second gives the desired 50 after which downsampling is safe.
    start = max(channel.start for channel in channels)
    channels = _match_sample_rates([channel[start:stop] for channel in channels], where="left")
    stop = min(channel.stop for channel in channels)
    channels = [channel[:stop] for channel in channels]
    if any(len(channel) == 0 for channel in channels):
        raise ValueError(message)
    if any(
        not np.array_equal(channel.timestamps, channels[0].timestamps) for channel in channels[1:]
    ):
        raise RuntimeError(
            "Provided force and trap position timestamps should match. "
            "Their sampling grids remain incompatible after cropping and rate matching. "
            "This can happen when channels were manually downsampled with centered timestamps. "
            "Pass the native trap-position and high-frequency force channels without manual "
            "downsampling; piezo tracking matches their sample rates automatically."
        )
    return channels


class ForceBaseLine:
    """A force baseline as a function of trap position"""

    def __init__(self, model, trap_data, force):
        """Force baseline

        Parameters
        ----------
        model : callable
            Model which returns the baseline at specified points.
        trap_data : Slice
            Trap mirror position data
        force : Slice
            Force data
        """
        self._model = model
        self._trap_data, self._force = _match_piezo_channels([trap_data, force])

    def valid_range(self):
        """Returns valid range of the baseline

        Returns the range of trap positions used to parameterize this baseline."""
        return np.min(self._trap_data.data), np.max(self._trap_data.data)

    def correct_data(self, force, trap_position):
        """Apply baseline correction to force data.

        Returns a slice with the baseline subtracted from the force data.

        Parameters
        ----------
        force : Slice
            Force data.
        trap_position : Slice
            Trap position data. Only complete samples in the shared time range are used.
        """
        trap_position, force = _match_piezo_channels([trap_position, force])

        return Slice(
            Continuous(
                force.data - self._model(trap_position.data),
                force._src.start,
                force._src.dt,
            ),
            labels={
                "title": force.labels.get("title", "Baseline Corrected Force"),
                "y": "Baseline Corrected Force (pN)",
            },
            calibration=force._calibration,
        )

    def plot(self, **kwargs):
        """Plot baseline fit

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.scatter()`.
        """
        import matplotlib.pyplot as plt

        plt.scatter(self._trap_data.data, self._force.data, s=2, **kwargs)
        plt.plot(self._trap_data.data, self._model(self._trap_data.data), "k")
        plt.xlabel("Mirror position")
        plt.ylabel(self._force.labels.get("y", "y"))
        plt.title("Force baseline")

    def plot_residual(self, **kwargs):
        """Plot residual

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.scatter`.
        """
        import matplotlib.pyplot as plt

        plt.scatter(
            self._trap_data.data,
            self._force.data - self._model(self._trap_data.data),
            s=2,
            **kwargs,
        )
        plt.xlabel("Mirror position")
        plt.ylabel(f"Residual {self._force.labels.get('y', 'y')}")
        plt.title("Fit residual")

    @classmethod
    def polynomial_baseline(cls, trap_position, force, degree=7, downsampling_factor=None):
        """Generate a polynomial baseline from data in the shared time range.

        Native channels are cropped and force is averaged to match the position
        sample rate. Prior manual downsampling is not required.

        Parameters
        ----------
        trap_position : Slice
            Trap mirror position data
        force : Slice
            Force data
        degree : int
            Polynomial degree
        downsampling_factor : int
            Factor by which to downsample before baseline determination
        """
        trap_position, force = _match_piezo_channels([trap_position, force])

        if downsampling_factor:
            trap_position, force = (
                ch.downsampled_by(downsampling_factor) for ch in (trap_position, force)
            )

        model = np.poly1d(np.polyfit(trap_position.data, force.data, deg=degree))
        return cls(model, trap_position, force)
