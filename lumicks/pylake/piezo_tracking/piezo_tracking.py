import warnings
import numpy as np
import matplotlib.pyplot as plt
from lumicks.pylake.channel import Slice


class DistanceCalibration:
    def __init__(self, trap_position, camera_distance, degree=1):
        """Map the trap position to the camera tracking distance using a linear fit.

        Parameters
        ----------
        trap_position : lumicks.pylake.Slice
            Trap position.
        camera_distance : lumicks.pylake.Slice
            Camera distance as determined by Bluelake.
            NOTE: The distance data should already have the bead diameter subtracted by Bluelake.
        degree : int
            Polynomial degree.
        """
        trap_position, camera_distance = trap_position.downsampled_like(camera_distance)
        mask = camera_distance.data != 0
        missed_frames = np.sum(1 - mask)
        if missed_frames > 0:
            warnings.warn(
                RuntimeWarning(
                    "There were frames with missing video tracking: "
                    f"{missed_frames} data point(s) were omitted."
                )
            )
        self.position, self.distance = trap_position.data[mask], camera_distance.data[mask]
        coeffs = np.polyfit(self.position, self.distance, degree)
        self._model = np.poly1d(coeffs)

    def __call__(self, trap_position):
        return Slice(
            trap_position._src._with_data(self._model(trap_position.data)),
            labels={"title": "Piezo distance", "y": "Distance [um]"},
        )

    def valid_range(self):
        return (np.min(self.position), np.max(self.position))

    def __str__(self):
        powers = np.flip(np.arange(self._model.order + 1))
        return "".join(
            f"{' + ' if coeff > 0 else ' - '}"
            f"{abs(coeff):.4f}"
            f"{'' if power == 0 else ' x' if power == 1 else f' x^{power}'}"
            for power, coeff in zip(powers, self._model.coeffs)
        ).strip()

    def __repr__(self):
        return f"DistanceCalibration({str(self)})"

    def plot(self):
        """Plot the calibration fit"""
        plt.scatter(self.position, self.distance, s=2, label="data")
        plt.plot(self.position, self._model(self.position), "k", label=f"${str(self)}$")
        plt.xlabel("Mirror position [um]")
        plt.ylabel("Camera Distance [um]")
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_residual(self):
        """Plot the residual of the calibration fit"""
        plt.scatter(self.position, self._model(self.position) - self.distance, s=2)
        plt.ylabel("Residual [um]")
        plt.xlabel("Mirror position [um]")
        plt.tight_layout()
        plt.legend()
        plt.show()

    @classmethod
    def from_file(cls, calibration_file, degree=1):
        """Use a reference measurement to calibrate trap mirror position to bead-bead distance.

        Parameters
        ----------
        calibration_file : pylake.File
        degree : int
            Polynomial order.
        """
        return cls(calibration_file["Trap position"]["1X"], calibration_file.distance1, degree)
