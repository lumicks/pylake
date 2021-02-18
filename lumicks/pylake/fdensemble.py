from lumicks.pylake.detail.alignment import align_fd_simple
import numpy as np


class FdEnsemble:
    """An ensemble of FD curves exported from Bluelake.

    This class provides a way to handle an ensemble of FD and perform procedures such as curve alignment on them.

    Attributes
    ----------
    fd_curves : Dict[lumicks.pylake.FdCurve]
        Dictionary of unprocessed FD curves.
    fd_curves_processed : Dict[lumicks.pylake.FdCurve]
        Dictionary of FD curves that were processed by the ensemble.

    Methods
    -------
    align_linear(distance_range_low, distance_range_high)
        Aligns F,d curves to the first F,d curve in the ensemble using two linear regressions.

    Examples
    --------
    ::

        import lumicks.pylake as lk

        file = lk.File("example.h5")
        fd_ensemble = lk.FdEnsemble(file.fdcurves)  # Create the ensemble

        # Use the first 0.02 and last 0.04 um of data to align the data
        fd_ensemble.align_linear(distance_range_low=0.02, distance_range_low=0.04)

        # Plot the aligned Fd curves
        for fd in fd_ensemble.values():
            fd.plot_scatter()
    """

    def __init__(self, fd_curves):
        self.fd_curves = fd_curves
        self.fd_curves_processed = fd_curves

    def __getitem__(self, item):
        return self.fd_curves_processed[item]

    def __iter__(self):
        return self.fd_curves_processed.__iter__()

    def items(self):
        return self.fd_curves_processed.items()

    def values(self):
        return self.fd_curves_processed.values()

    def keys(self):
        return self.fd_curves_processed.keys()

    @property
    def raw(self):
        return self.fd_curves

    @property
    def f(self):
        return np.hstack([fd.f.data for fd in self.values()])

    @property
    def d(self):
        return np.hstack([fd.d.data for fd in self.values()])

    def align_linear(self, distance_range_low, distance_range_high):
        """Aligns F,d curves to the first F,d curve in the ensemble.

        Force is aligned by taking the mean of the lowest distances. Distance is aligned by considering the last segment
        of each F,d curve. This method regresses a line to the last segment of each F,d curve and aligns the curves
        based on this regressed line. Note that this requires the ends of the aligned F,d curves to be in a comparably
        folded state and obtained in the elastic range of the force, distance curve. If any of these assumptions are not
        met, this method should not be applied.

        Parameters
        ----------
        distance_range_low : float
            Range of distances to use for the force alignment. Distances in the range [smallest_distance,
            smallest_distance + distance_range_low) are used to determine the force offsets.
        distance_range_high : float
            Upper range of distances to use. Distances in the range [largest_distance - distance_range_high,
            largest_distance] are used for the distance alignment."""
        self.fd_curves_processed = align_fd_simple(
            self.fd_curves, distance_range_low, distance_range_high
        )
