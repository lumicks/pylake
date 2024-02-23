import numpy as np


def _guess_background(data):
    """Determine background level by determining most prominent mode"""
    import scipy.stats

    kde_estimate = scipy.stats.gaussian_kde(data, 0.02)
    interpolated_kde = np.arange(min(data), np.max(data))
    return interpolated_kde[np.argmax(kde_estimate.pdf(interpolated_kde))]


def _move_inward_percentile(data, bead_edges, threshold_percentile):
    """Move inward until a specific percentile is reached"""

    final_cutoff = np.percentile(data[slice(*bead_edges)], threshold_percentile)
    return [
        bead_edges[0] + np.where(data[bead_edges[0] :] <= final_cutoff)[0][0],
        bead_edges[1] - np.where(np.flip(data[: bead_edges[1]]) <= final_cutoff)[0][0],
    ]


def find_beads_brightness(
    kymograph_image,
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
    kymograph_image : np.ndarray
        2D kymograph image
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
    bead_edges : list[int]
       List of the two edge positions in pixels.

    Raises
    ------
    RuntimeError
        When the algorithm fails to locate two edges during the bead finding stage.
    """
    import scipy.ndimage
    import matplotlib.pyplot as plt

    summed_kymo = kymograph_image.sum(axis=1)
    data = summed_kymo.astype(float)

    # Makes sure we can handle dark beads
    if allow_negative_beads:
        # Try to guess the background by finding the most prominent mode
        baseline = _guess_background(data)
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

    bead_edges = _move_inward_percentile(data, bead_edges, threshold_percentile)

    if plot:
        [plt.axvline(be, label="edge", color="k", linestyle="--") for be in bead_edges]
        plt.legend()

    return bead_edges


def find_beads_template(
    kymograph_image,
    bead_diameter_pixels,
    downsample_num_frames=5,
    plot=False,
    allow_movement=False,
    threshold_percentile=70,
):
    """Determine bead edges using a cross correlation procedure

    This function seeks a bead-sized template that appears in subsequent frames.
    It then moves inward by half the template size to provide bead edges.

    Parameters
    ----------
    kymograph_image : np.ndarray
        2D kymograph image
    bead_diameter_pixels : int
        Template size in pixels (will be rounded up to an odd number). Note that
        the template size should be equal or larger than the bead size. It should
        capture as much of the bead (including the fluorescent side lobe if present)
        as possible.
    downsample_num_frames : int
        Number of time frames to downsample to (must be larger than 3).
    plot : bool
        Plots results that can be used for debugging why it has failed.
    allow_movement : bool
        Allow movement of the template between frames?
    threshold_percentile : int
        Percentile below which to drop.

    Returns
    -------
    bead_edges : list[int]
       List of the two edge positions in pixels.

    Raises
    ------
    ValueError
        If downsample_num_frames is set to a value lower than 3.
    ValueError
        If the kymograph image is shorter than 3 frames or has incorrect dimensions.
    """
    import scipy.ndimage
    import matplotlib.pyplot as plt
    from skimage.measure import block_reduce

    if downsample_num_frames < 3:
        raise ValueError(
            f"Time axis needs to be divided in at least 3 sections, provided {downsample_num_frames}"
        )

    if kymograph_image.ndim != 2:
        raise ValueError(
            f"Kymograph image must be two dimensional. Provided {kymograph_image.ndim} dimensional "
            "image"
        )

    if kymograph_image.shape[1] < 3:
        raise ValueError(
            "Kymograph is too short to apply this method. This method requires at least 3 scan "
            f"lines while this kymograph has {kymograph_image.shape[1]} scan lines."
        )

    half_width = int(bead_diameter_pixels // 2)
    template_size_pixels = half_width * 2 + 1

    ds_factor = kymograph_image.shape[1] // downsample_num_frames
    img = (
        block_reduce(kymograph_image, (1, ds_factor), func=np.sum)[
            :, : kymograph_image.shape[1] // ds_factor
        ]
        if ds_factor > 1
        else kymograph_image
    )

    # Try to guess the background by finding the most prominent mode
    baseline = _guess_background(img.flatten())

    # Flip dark beads into having positive signal that we can template match on
    img = np.abs(img - baseline)

    # Pad so that we don't get boundary effects from our filtering
    padding = np.zeros((half_width * 2, img.shape[1]))
    padded_img = np.vstack((padding, img, padding))

    # Set up the correlation matrix. For each frame, we apply a sliding window.
    # for each window we look into the subsequent frames whether we see the window there
    # using cross correlation.
    if allow_movement:

        def normalized_cross_correlation(template_origin, t):
            template = padded_img[template_origin : template_origin + template_size_pixels, t]
            normalization_factor = 1.0 + np.std(padded_img[:, t + 1]) * np.std(template)

            return (
                np.max(scipy.signal.correlate(padded_img[:, t + 1], template, mode="same"))
                / normalization_factor
            )

        correlation_stack = np.stack(
            [
                [
                    normalized_cross_correlation(template_origin, t)
                    for t in range(padded_img.shape[1] - 1)
                ]
                for template_origin in range(padded_img.shape[0])
            ]
        )
    else:

        def normalized_cross_correlation_row(current, subsequent, row_std):
            mul = current * subsequent
            template_mean = scipy.ndimage.uniform_filter(current, template_size_pixels)
            template_sq = scipy.ndimage.uniform_filter(current**2, template_size_pixels)
            # Sometimes this nets tiny negative values, hence the absolute value
            template_std = np.sqrt(np.abs(template_sq - template_mean**2))

            normalization = 1.0 + template_std * row_std
            return scipy.ndimage.uniform_filter(mul, template_size_pixels) / normalization

        row_stds = np.std(padded_img, axis=0)
        correlation_stack = np.stack(
            [
                normalized_cross_correlation_row(current, subsequent, row_std)
                for (current, subsequent, row_std) in zip(
                    padded_img[:, :-1].T, padded_img[:, 1:].T, row_stds[1:]
                )
            ]
        )[
            :, template_size_pixels // 2 : -template_size_pixels // 2
        ].T  # remove overhangs

    # Only use positive correlation
    positive_correlation = correlation_stack * (correlation_stack > 0)

    # Sum over time
    aggregate = positive_correlation.sum(axis=1)

    # If the feature we are looking for is smaller than the template, the feature position with
    # respect to the template becomes arbitrary.
    #
    #  ___/\__ will have just as good a score as /\_____
    #
    # To instill a preference for centering the template, we blur the match score such that this
    # flat match score now features a peak in the middle that we can find.
    aggregate = scipy.ndimage.gaussian_filter(aggregate, half_width // 4)
    aggregate = aggregate[half_width:-half_width]  # Remove the padding

    # Find local positive non-zero maxima
    peaks = (
        scipy.ndimage.grey_dilation([aggregate], (0, 2 * template_size_pixels)) == aggregate
    ).flatten()
    peaks *= aggregate > 0
    peak_centers = np.where(peaks)[0]

    if plot:
        # Show the correlation map
        plt.figure()
        plt.imshow(positive_correlation, aspect="auto")

        for m in peak_centers:
            plt.axhline(m + half_width, color="k", linestyle="--")

        plt.figure()
        plt.plot(np.arange(len(aggregate)), aggregate, label="correlation score")
        plt.twinx()
        plt.plot(img[:, 0], "k", alpha=0.5, label="intensities")
        plt.plot(img[:, 1:], "k", alpha=0.5, label="_")

        if np.any(peak_centers):
            plt.axvline(peak_centers[0], label="center")
            if len(peak_centers) > 0:
                for m in peak_centers[1:]:
                    plt.axvline(m, label="_")

    if len(peak_centers) < 2:
        raise RuntimeError("Did not find two beads")

    # Seek inward until we cross the baseline.
    bead_edges = [peak_centers[0] + half_width, peak_centers[-1] - half_width]
    data = np.sum(img, axis=1)
    bead_edges = _move_inward_percentile(data, bead_edges, threshold_percentile)

    if plot:
        plt.axvline(bead_edges[0], label="edge", color="k")
        plt.axvline(bead_edges[1], label="_edge", color="k")

    return bead_edges
