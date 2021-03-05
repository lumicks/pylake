import numpy as np


def calculate_msd(frame_idx, position, max_lag):
    """Estimate the Mean Square Displacement (MSD) for various time lags.

    The estimator for the MSD (rho) is defined as:

      rho_n = (1 / (N-n)) sum_{i=1}^{N-n}(r_{i+n} - r_{i})^2

    here N refers to the total frames, n to the lag time and r_i the spatial position at lag i.
    This function produces a list of lag times and mean squared displacements for those lag times.

    Parameters
    ----------
    frame_idx : array_like
        List of frame indices (note that these have to be of integral type to prevent rounding
        errors).
    position : array_like
        List of positions.
    max_lag : float
        Maximum lag to include (note that MSD estimates generally do not get better by including
        several lag steps).
    """
    frame_mesh_1, frame_mesh_2 = np.meshgrid(frame_idx, frame_idx)
    frame_diff = frame_mesh_1 - frame_mesh_2
    frame_lags = np.unique(frame_diff)

    position_mesh_1, position_mesh_2 = np.meshgrid(position, position)
    summand = (position_mesh_1 - position_mesh_2) ** 2

    # Look up only the rho elements we need
    frame_lags = frame_lags[frame_lags > 0][:max_lag]
    msd = [np.mean(summand[frame_diff == delta_frame]) for delta_frame in frame_lags]

    return frame_lags, msd
