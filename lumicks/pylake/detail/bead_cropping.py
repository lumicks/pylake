import numpy as np


def find_beads(
    summed_kymo,
    bead_diameter_pixels,
    plot=False,
    threshold_percentile=70,
    allow_negative_beads=True,
):
    """Find bead edges in kymograph image sum

    Attempts to search for the bead edges by checking where the fluorescence drops below a
    threshold. Only intended for stationary beads.

    Parameters
    ----------
    summed_kymo : np.ndarray
        Kymograph image data (single channel summed along the time axis).
    bead_diameter_pixels : float
        Estimate for the bead size in pixels.
    plot : bool
        Plot result
    threshold_percentile : int
        Percentile below which to drop.
    allow_negative_beads : bool
        Tries to estimate a background from the dominant mode of the photon counts. This estimate
        is subtracted from the raw counts prior to analysis.

    Returns
    -------
    list of int
       List of the two edge positions in pixels.

    Raises
    ------
    RuntimeError
        When the algorithm fails to locate two edges during the bead finding stage.
    """
    import scipy.ndimage
    import matplotlib.pyplot as plt

    data = summed_kymo.astype(float)

    # Makes sure we can handle dark beads
    if allow_negative_beads:
        # Try to guess the background by finding the most prominent mode
        kde_estimate = scipy.stats.gaussian_kde(data, 0.02)
        interpolated_kde = np.arange(min(data), np.max(data))
        baseline = interpolated_kde[np.argmax(kde_estimate.pdf(interpolated_kde))]
        data = np.abs(data - baseline)
    else:
        baseline = 0

    # Get rid of small fluorescence peaks.
    only_beads = scipy.ndimage.grey_opening(data, int(bead_diameter_pixels / 4) * 2 + 1)

    # Get rough estimate of where the beads are.
    blurred, blurred_derivative = (
        scipy.ndimage.gaussian_filter(
            only_beads, bead_diameter_pixels, mode="constant", cval=0, order=order
        )
        for order in (0, 1)
    )

    # Find negative zero crossings.
    bead_edges = np.where(np.diff(np.sign(blurred_derivative)) == -2)[0]

    if plot:
        plt.plot(summed_kymo, label="input data")
        plt.axhline(baseline, label="baseline")
        plt.plot(data, label="transformed data")
        plt.plot(only_beads, label="only beads")
        plt.plot(blurred, label="Gaussian filtered")

        for be in bead_edges:
            plt.axvline(be, label="center guess", color="k", linestyle="--", alpha=0.5)

        plt.legend()  # In case we don't make it past the exception

    if len(bead_edges) != 2:
        raise RuntimeError("Did not find two beads")

    # We move inwards until the opened trace drops below the Gaussian blurred trace. This will be
    # roughly halfway on the slope.
    bead_edges[0] += np.where(only_beads[bead_edges[0] :] < blurred[bead_edges[0] :])[0][0]
    bead_edges[1] -= np.where(
        np.flip(only_beads[: bead_edges[1]]) < np.flip(blurred[: bead_edges[1]])
    )[0][0]

    # We then keep moving until we've hit a low enough signal level that we are confident we
    # have reached the tether.
    final_cutoff = np.percentile(data[bead_edges[0] : bead_edges[1]], threshold_percentile)
    bead_edges[0] += np.where(data[bead_edges[0] :] <= final_cutoff)[0][0]
    bead_edges[1] -= np.where(np.flip(data[: bead_edges[1]]) <= final_cutoff)[0][0]

    if plot:
        [plt.axvline(be, label="edge", color="k", linestyle="--") for be in bead_edges]
        plt.legend()

    return bead_edges
