import numpy as np
import matplotlib.pyplot as plt
from lumicks.pylake.channel import Slice, Continuous


class ForceBaseLine:
    def __init__(self, model, trap_data, force):
        """Force baseline

        Parameters
        ----------
        model : callable
            Model which returns the baseline at specified points.
        trap_data : lumicks.pylake.Slice
            Trap mirror position data
        force : lumicks.pylake.Slice
            Force data
        """
        self._model = model
        self._trap_data = trap_data
        self._force = force

    def valid_range(self):
        """Returns valid range of the baseline

        Returns the range of trap positions used to parameterize this baseline."""
        return np.min(self._trap_data.data), np.max(self._trap_data.data)

    def correct_data(self, force, trap_position):
        """Apply baseline correction to force data.

        Returns a slice with the baseline subtracted from the force data.

        Parameters
        ----------
        force : lumicks.pylake.Slice
            Force data.
        trap_position : lumicks.pylake.Slice
            Trap position data (needs to be the same time range as the force data).
        """

        if not np.array_equal(force.timestamps, trap_position.timestamps):
            raise RuntimeError("Provided force and trap position timestamps should match")

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

    def plot(self):
        plt.scatter(self._trap_data.data, self._force.data, s=2)
        plt.plot(self._trap_data.data, self._model(self._trap_data.data), "k")
        plt.xlabel("Mirror position")
        plt.ylabel(self._force.labels["y"])
        plt.title("Force baseline")

    def plot_residual(self):
        plt.scatter(self._trap_data.data, self._force.data - self._model(self._trap_data.data), s=2)
        plt.xlabel("Mirror position")
        plt.ylabel(f"Residual {self._force.labels['y']}")
        plt.title("Fit residual")

    @classmethod
    def polynomial_baseline(cls, trap_position, force, degree=7, downsampling_factor=None):
        """Generate a polynomial baseline from data

        Parameters
        ----------
        trap_position : lumicks.pylake.Slice
            Trap mirror position data
        force : lumicks.pylake.Slice
            Force data
        degree : int
            Polynomial degree
        downsampling_factor : int
            Factor by which to downsample before baseline determination
        """
        if not np.array_equal(force.timestamps, trap_position.timestamps):
            raise RuntimeError("Provided force and trap position timestamps should match")

        if downsampling_factor:
            trap_position, force = (
                ch.downsampled_by(downsampling_factor) for ch in (trap_position, force)
            )

        model = np.poly1d(np.polyfit(trap_position.data, force.data, deg=degree))
        return cls(model, trap_position, force)