import warnings

import numpy as np

from ..channel import Slice
from .baseline import ForceBaseLine

__all__ = ["DistanceCalibration", "PiezoTrackingCalibration", "PiezoForceDistance"]


class DistanceCalibration:
    """Class to calibrate trap position to camera distance"""

    def __init__(self, trap_position, camera_distance, degree=1):
        """Map the trap position to the camera tracking distance using a linear fit.

        Parameters
        ----------
        trap_position : Slice
            Trap position.
        camera_distance : Slice
            Camera distance as determined by Bluelake.
            NOTE: The distance data should already have the bead diameter subtracted by Bluelake.
        degree : int
            Polynomial degree.
        """
        trap_position, camera_distance = trap_position.downsampled_like(camera_distance)
        mask = camera_distance.data != 0
        self.position, self.distance = trap_position.data[mask], camera_distance.data[mask]

        missed_frames = camera_distance.data.size - self.distance.size
        if missed_frames > 0:
            warnings.warn(
                RuntimeWarning(
                    "There were frames with missing video tracking: "
                    f"{missed_frames} data point(s) were omitted."
                )
            )

        coeffs = np.polyfit(self.position, self.distance, degree)
        self._model = np.poly1d(coeffs)

    def __call__(self, trap_position):
        return Slice(
            trap_position._src._with_data(self._model(trap_position.data)),
            labels={"title": "Piezo distance", "y": "Distance [um]"},
        )

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
        import matplotlib.pyplot as plt

        plt.scatter(self.position, self.distance, s=2, label="data")
        plt.plot(self.position, self._model(self.position), "k", label=f"${str(self)}$")
        plt.xlabel("Mirror position")
        plt.ylabel("Camera Distance [um]")
        plt.tight_layout()
        plt.legend()

    def plot_residual(self):
        """Plot the residual of the calibration fit"""
        import matplotlib.pyplot as plt

        plt.scatter(self.position, self._model(self.position) - self.distance, s=2)
        plt.ylabel("Residual [um]")
        plt.xlabel("Mirror position")
        plt.tight_layout()

    @classmethod
    def from_file(cls, calibration_file, degree=1):
        """Use a reference measurement to calibrate trap mirror position to bead-bead distance.

        Parameters
        ----------
        calibration_file : lumicks.pylake.File
        degree : int
            Polynomial order.
        """
        return cls(calibration_file["Trap position"]["1X"], calibration_file.distance1, degree)


class PiezoTrackingCalibration:
    """Class to handle piezo tracking calibration

    Allows calculating piezo distance from trap position and correlated force data"""

    def __init__(
        self,
        trap_calibration,
        signs=(1, -1),
    ):
        """Set up piezo tracking calibration

        trap_calibration : DistanceCalibration
            Calibration from trap position to trap to trap distance.
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
        self._signs = signs

    def piezo_track(self, trap_position, force1, force2, downsampling_factor=None):
        """Obtain piezo distance and baseline corrected forces

        Parameters
        ----------
        trap_position : Slice
            Trap position.
        force1 : Slice
            First force channel to use for piezo tracking.
        force2 : Slice
            Second force channel to use for piezo tracking.
        downsampling_factor : Optional[int]
            Downsampling factor.
        """
        if downsampling_factor:
            trap_position, force1, force2 = (
                x.downsampled_by(downsampling_factor) for x in (trap_position, force1, force2)
            )

        trap_trap_dist = self.trap_calibration(trap_position)
        bead_displacements = 1e-3 * sum(
            sign * force / force.calibration[0]["kappa (pN/nm)"]
            for force, sign in zip((force1, force2), self._signs)
        )

        piezo_distance = trap_trap_dist - bead_displacements

        return piezo_distance


class PiezoForceDistance:
    """Class to determine both piezo distance and baseline corrected force"""

    def __init__(
        self,
        trap_calibration,
        baseline_force1=None,
        baseline_force2=None,
        signs=(1, -1),
    ):
        """Set up piezo force distance data

        trap_calibration : DistanceCalibration
            Calibration from trap position to trap to trap distance.
        baseline_force1 : ForceBaseline
            Baseline for force1 (optional)
        baseline_force2 : ForceBaseline
            Baseline for force2 (optional)
        signs : tuple(float, float)
            Sign convention for forces (e.g. (1, -1) indicates that force2 is negative).
        """
        for argument, variable, instance_type in zip(
            ("first", "second", "third"),
            (trap_calibration, baseline_force1, baseline_force2),
            (DistanceCalibration, ForceBaseLine, ForceBaseLine),
        ):
            if variable is not None and not isinstance(variable, instance_type):
                raise TypeError(
                    f"Expected {instance_type.__name__} for the {argument} argument, "
                    f"got {type(variable).__name__}"
                )

        self.piezo_calibration = PiezoTrackingCalibration(trap_calibration, signs)
        self.baseline_force1 = baseline_force1
        self.baseline_force2 = baseline_force2
        self._signs = signs

    def valid_range(self):
        """Returns the mirror position range in which the piezo tracking is valid"""
        calibration_items = (self.baseline_force1, self.baseline_force2)
        ranges = [r.valid_range() for r in calibration_items if r]
        if not ranges:
            return -np.inf, np.inf

        stacked_ranges = np.stack(ranges)
        return np.max(stacked_ranges[:, 0]), np.min(stacked_ranges[:, 1])

    def force_distance(self, trap_position, force1, force2, trim=True, downsampling_factor=None):
        """Obtain piezo distance and baseline corrected forces

        Parameters
        ----------
        trap_position : Slice
            Trap position.
        force1 : Slice
            First force channel to use for piezo tracking.
        force2 : Slice
            Second force channel to use for piezo tracking.
        trim : bool
            Trim regions outside the calibration range.
        downsampling_factor : Optional[int]
            Downsampling factor.
        """
        piezo_distance = self.piezo_calibration.piezo_track(
            trap_position, force1, force2, downsampling_factor
        )

        if downsampling_factor:
            trap_position, force1, force2 = (
                x.downsampled_by(downsampling_factor) for x in (trap_position, force1, force2)
            )

        corrected_forces = [
            baseline.correct_data(force, trap_position) if baseline else force
            for baseline, force in zip(
                (self.baseline_force1, self.baseline_force2), (force1, force2)
            )
        ]

        if trim:
            valid_range = self.valid_range()
            valid_mask = np.logical_and(
                valid_range[0] <= trap_position.data, trap_position.data <= valid_range[1]
            )
            piezo_distance = piezo_distance[valid_mask]
            corrected_forces = (channel_slice[valid_mask] for channel_slice in corrected_forces)

        return (piezo_distance, *corrected_forces)
