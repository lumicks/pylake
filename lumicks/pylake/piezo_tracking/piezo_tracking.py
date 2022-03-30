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
        plt.xlabel("Mirror position")
        plt.ylabel("Camera Distance [um]")
        plt.tight_layout()
        plt.legend()

    def plot_residual(self):
        """Plot the residual of the calibration fit"""
        plt.scatter(self.position, self._model(self.position) - self.distance, s=2)
        plt.ylabel("Residual [um]")
        plt.xlabel("Mirror position")
        plt.tight_layout()
        plt.legend()

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


class PiezoTrackingCalibration:
    def __init__(
        self,
        trap_calibration,
        baseline_force1,
        baseline_force2,
        signs=(1, -1),
    ):
        """Set up piezo tracking calibration

        trap_calibration : pylake.DistanceCalibration
            Calibration from trap position to trap to trap distance.
        baseline_force1 : pylake.ForceBaseline
            Baseline for force1
        baseline_force2 : pylake.ForceBaseline
            Baseline for force2
        signs : tuple(float, float)
            Sign convention for forces (e.g. (1, -1) indicates that force2 is negative).
        """
        if len(signs) != 2:
            raise ValueError(
                "Argument `signs` should be a tuple of two floats reflecting the sign for each "
                "channel."
            )
        for sign in signs:
            if abs(sign) != 1:
                raise ValueError("Each sign should be either -1 or 1.")

        self.trap_calibration = trap_calibration
        self.baseline_force1 = baseline_force1
        self.baseline_force2 = baseline_force2
        self._signs = signs

    def valid_range(self):
        """Returns the mirror position range in which the piezo tracking is valid"""
        calibration_items = (self.trap_calibration, self.baseline_force1, self.baseline_force2)
        return np.min(np.stack([r.valid_range() for r in calibration_items]), axis=0)

    def piezo_track(self, trap_position, force1, force2, trim=True):
        """Obtain piezo distance and baseline corrected forces

        Parameters
        ----------
        trap_position : pylake.channel.Slice
            Trap position.
        force1 : pylake.channel.Slice
            First force channel to use for piezo tracking.
        force2 : pylake.channel.Slice
            Second force channel to use for piezo tracking.
        trim : bool
            Trim regions outside the calibration range.
        """
        trap_trap_dist = self.trap_calibration(trap_position)
        bead_displacements = 1e-3 * sum(
            sign * force / force.calibration[0]["kappa (pN/nm)"]
            for force, sign in zip((force1, force2), self._signs)
        )
        piezo_distance = trap_trap_dist - bead_displacements
        corrected_force1 = self.baseline_force1.correct_data(force1, trap_position)
        corrected_force2 = self.baseline_force2.correct_data(force2, trap_position)

        if trim:
            valid_range = self.valid_range()
            valid_mask = np.logical_and(
                valid_range[0] <= trap_position.data, trap_position.data <= valid_range[1]
            )
            piezo_distance, corrected_force1, corrected_force2 = (
                piezo_distance[valid_mask],
                corrected_force1[valid_mask],
                corrected_force2[valid_mask],
            )

        return piezo_distance, corrected_force1, corrected_force2
