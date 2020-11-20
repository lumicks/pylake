import numpy as np


def generate_clip_window(line_length, peak_positions, half_size):
    """Generates a clipping window for fitting Gaussian peaks

    Basically it provides a window for which data points to include in the fit. Only regions around the peaks are used.
    Since peaks can overlap but share the same background, we do want to fit multiple peaks at once. However, since
    users may deliberately omit traces or other artifacts (such as beads) from the fit, some of the data needs to be
    rejected.

    Parameters
    ----------
    line_length : int
        length of a kymograph line
    peak_positions : array_like
        List of peak positions obtained from a previous estimate
    half_size : int
        The data length used to refine the line estimate will be 2 * half_size + 1.
    """
    mask = np.zeros(line_length, dtype=bool)
    for pos in peak_positions:
        lb = np.clip(int(np.floor(pos - half_size)), 0, line_length)
        ub = np.clip(int(np.ceil(pos + half_size + 1)), 0, line_length)
        mask[lb: ub] = True

    return mask

