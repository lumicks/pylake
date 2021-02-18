import numpy as np
from lumicks.pylake.detail.utilities import first


def align_force_simple(fd_curves, distance_range=1):
    """Aligns F,d curves to the first F,d curves in the ensemble by force

    It evaluates the difference in offset by considering the mean of the early data points. The underlying assumption
    is that the force does not increase substantially for the first part of the pulling curve and that curves have
    been acquired starting from the same F,d point."""
    assert len(fd_curves) > 1, "Alignment only makes sense for more than 1 curve"

    def get_offset(fd):
        force, distance = fd._sliced(distance_max=np.min(fd.d.data[fd.d.data > 0]) + distance_range)
        return np.mean(force)

    reference_offset = get_offset(first(fd_curves.values()))
    return {
        key: fd.with_offset(force_offset=reference_offset - get_offset(fd))
        for key, fd in fd_curves.items()
    }


def align_distance_simple(fd_curves, distance_range=1):
    """Aligns F,d curves to the first F,d curve in the ensemble by distance. Note that force has to be aligned first.

    This method regresses a line to the last segment of each F,d curve and aligns the curves based on this regressed
    line. Note that this requires the ends of the aligned F,d curves to be in a comparably folded state and obtained
    in the elastic range of the force, distance curve. If any of these assumptions are not met, this method should not
    be applied."""
    assert len(fd_curves) > 1, "Alignment only makes sense for more than 1 curve"

    def linear_fit(fd):
        force, distance = fd._sliced(distance_min=np.max(fd.d.data) - distance_range)
        poly_coefficients = np.polyfit(distance, force, 1)
        return poly_coefficients[1], poly_coefficients[0]

    ref_offset, ref_slope = linear_fit(first(fd_curves.values()))

    def get_offset(fd):
        force, distance = fd._sliced(distance_min=np.max(fd.d.data) - distance_range)
        force_sim = ref_offset + ref_slope * distance
        dx = np.mean(force - force_sim) / ref_slope
        return dx

    return {key: fd.with_offset(distance_offset=get_offset(fd)) for key, fd in fd_curves.items()}


def align_fd_simple(fd_curves, distance_range_low, distance_range_high):
    """Aligns F,d curves to the first F,d curve in the ensemble.

    Force is aligned by taking the mean of the lowest distances. Distance is aligned by considering the last segment
    of each F,d curve. This method regresses a line to the last segment of each F,d curve and aligns the curves based
    on this regressed line. Note that this requires the ends of the aligned F,d curves to be in a comparably folded
    state and obtained in the elastic range of the force, distance curve. If any of these assumptions are not met, this
    method should not be applied.

    Parameters
    ----------
    fd_curves : Dict[lumicks.pylake.FdCurve]
        List of F,d curves to apply the method to.
    distance_range_low : float
        Range of distances to use for the force alignment. Distances in the range [smallest_distance,
        smallest_distance + distance_range_low) are used to determine the force offsets.
    distance_range_high : float
        Upper range of distances to use. Distances in the range [largest_distance - distance_range_high,
        largest_distance] are used for the distance alignment."""
    return align_distance_simple(
        align_force_simple(fd_curves, distance_range_low), distance_range_high
    )
