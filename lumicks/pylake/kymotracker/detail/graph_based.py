from lumicks.pylake.kymotracker.detail.peakfinding import KymoPeaks, peak_estimate, refine_peak_based_on_moment
import numpy as np


def find_peaks(kymograph_data, half_width_pixels, pixel_threshold, bias_correction=True, rect=None):
    coordinates, time_points = peak_estimate(kymograph_data, half_width_pixels, pixel_threshold)
    if len(coordinates) == 0:
        return KymoTrackGroup([])

    position, time, m0 = refine_peak_based_on_moment(
        kymograph_data,
        coordinates,
        time_points,
        half_width_pixels,
        bias_correction=bias_correction,
    )

    if rect:
        (t0, p0), (t1, p1) = _to_pixel_rect(
            rect, kymograph.pixelsize[0], kymograph.line_time_seconds
        )
        mask = (position >= p0) & (position < p1) & (time >= t0) & (time < t1)
        position, time, m0 = position[mask], time[mask], m0[mask]

        if len(position) == 0:
            return KymoTrackGroup([])

    return KymoPeaks(position, time, m0)


def calculate_costs(current_frame, next_frame, max_distance):
    """Calculates the cost function between pairs of frames"""
    positional_cost = (current_frame.coordinates[:, np.newaxis] - next_frame.coordinates) ** 2
    positional_cost[positional_cost > max_distance * max_distance] = np.inf
    intensity_cost = (current_frame.coordinates[:, np.newaxis] - next_frame.coordinates) ** 2

    # Add the dummy row and column
    cost = np.full(np.array(positional_cost.shape) + 1, max_distance * max_distance, dtype=float)
    cost[1:, 1:] = positional_cost

    return cost


def initialize_positions(costs):
    """Provides the initial linking matrix in coordinate form

    Parameters
    ----------
    costs : numpy.ndarray
        2D array of cost functions representing the cost of linking
        particle i in frame f1, to particle j in frame f2

    Returns
    -------
    current_to_next : numpy.ndarray
        Which particle this one connects to. The zeroth index is
        reserved for the dummy particle (which means not connected
        to anything).
    next_to_current : numpy.ndarray
        Which particle this one is connected from. The zeroth index
        is reserved for the dummy particle (which means not connected
        to anything).
    """

    # Make our own internal copy so we can modify it.
    costs = np.copy(costs)

    # In this initialization, we don't want to explicitly link anything
    # to the dummy costs (since that is already the default) so we
    # set those to infinite.
    value = 0

    # Negative indicates unlinked
    current_to_next = np.full((costs.shape[0],), 0, dtype=int)
    next_to_current = np.full((costs.shape[1],), 0, dtype=int)
    while value < np.inf:
        (i, j) = np.unravel_index(np.argmin(costs), costs.shape)
        value = costs[i, j]

        costs[i, :] = np.inf
        costs[:, j] = np.inf
        current_to_next[i] = j
        next_to_current[j] = i

    return current_to_next, next_to_current


def optimize_linking(initial_condition, cost, null_cost, max_iter=100):
    """Optimize linking of particles between frames.

    This swaps which particles are linked to minimize the cost. As long as the cost
    is a simple function that only depends on the particle indices themselves, this
    is guaranteed to converge to a global optimum.

    Parameters
    ----------
    initial_condition : tuple of numpy.ndarray
        Set of coordinate pairs with from and to coordinates. A negative
        index indicates unlinked.
    cost : numpy.ndarray
        2D array of cost functions representing the cost of linking
        particle i in frame f1, to particle j in frame f2
    null_cost : numpy.ndarray
        The cost of making a particle appear or disappear (connection_tolerance * frame_difference).
    max_iter : int
        Maximum number of iterations.
    """

    # We try to set a particular node in the adjacency matrix to 1 and see if this reduces the
    # overall cost. If we set g_{i,j} = 1 then there are a few cases to distinguish:
    #
    # 1. There's already a particle relation related from i (i, l) and
    #    to j (k, j), just not from i to j in which case we reconnect them:
    #
    #      g{i, l} = g{k, j} will be set to 0
    #      g{k, l} will be set to 1.
    #
    #    In this case, the score change is:
    #
    #      delta_cost = cost[i, j] + cost[k, l] - cost[i, l] - cost[k, j]
    #
    # 2. A particle appears at j whereas it previously had a relation to something in the previous frame (k)
    #
    #      delta_cost = cost[null, j] + cost[k, null] - cost[k, j]
    #
    # 3. A particle disappears from at i whereas it previously had a relation to something in the next frame (l)
    #
    #      delta_cost = cost[i, null] + cost[null, l] - cost[i, l]
    #
    # Because we use these particular rules, the graph topology never changes.

    # Particle linkages
    # Storing them this way allows for quicker lookups at the price of setting a few more elements,
    # when we actually make a switch (rare). Otherwise we'd have to find nonzero's all the time.
    current_to_next, next_to_current = initial_condition

    # Unfortunately, inherently sequential sweeps, since they need to be updated immediately.
    # TODO: Likely candidate for jit treatment.
    for _ in range(max_iter):
        # Tries to add a particle linkage between two particle positions.
        # Note that this requires swapping their other associations to ensure that the
        # topology of the graph stays the same (1 to 1 except for the dummy particle which can
        # have 1 to many links).
        lowest_cost = 0
        lowest_change = None
        for i in range(1, cost.shape[0]):
            for j in range(1, cost.shape[1]):
                # Only consider adding ones that have a finite cost and don't already exist
                if cost[i, j] < np.inf and not current_to_next[i] == j:
                    # We're reassigning particles that already had a connection
                    #   i->l, k->j  =>  i->j k->l
                    k = next_to_current[j]
                    l = current_to_next[i]
                    cost_change = cost[i, j] + cost[k, l] - cost[i, l] - cost[k, j]
                    if cost_change < lowest_cost and np.isfinite(cost_change):
                        lowest_cost = cost_change
                        lowest_change = (i, j, k, l)

        # Particles disappearing (j becomes 0)
        for i in range(1, cost.shape[0]):
            #   i->j => i->0
            k = 0
            l = current_to_next[i]
            cost_change = cost[i, 0] + cost[0, l] - cost[i, l]
            if cost_change < lowest_cost and np.isfinite(cost_change):
                lowest_cost = cost_change
                lowest_change = (i, j, k, l)

        # Particles appearing (i becomes 0)
        for j in range(1, cost.shape[1]):
            #   i->j => 0->j
            k = next_to_current[j]
            l = 0
            cost_change = cost[0, j] + cost[k, 0] - cost[k, j]
            if cost_change < lowest_cost and np.isfinite(cost_change):
                lowest_cost = cost_change
                lowest_change = (i, j, k, l)

        if lowest_cost < 0:
            # Relink particles
            #   i->l, k->j  =>  i->j k->l
            i, j, k, l = lowest_change
            next_to_current[j], current_to_next[i], next_to_current[l], current_to_next[
                k] = lowest_change
        else:
            # Done!
            break

    extracted_costs = [cost[idx, target] for idx, target in enumerate(current_to_next)]

    return current_to_next, next_to_current, extracted_costs


max_distance = 10


def calculate_associations(from_frame, to_frame, max_distance):
    cost_matrix = calculate_costs(from_frame, to_frame, max_distance)
    initial_guess = initialize_positions(cost_matrix)
    to_next, to_current, connection_cost = optimize_linking(initial_guess, cost_matrix,
                                                            max_distance)
    return to_next, connection_cost


kymo2 = f.kymos["1"]
data = kymo["10s":"160s"].crop_by_distance(8.5, 20).get_image("green")

half_width = 3
thresh = 4

peaks = find_peaks(data, half_width, thresh, bias_correction=True)

# Track assignment
# We make a structure parallel to `KymoPeaks` that stores which track is assigned to which peak.
# This allows us to look these up quickly.
track_assignments = [np.full(frame.coordinates.shape, -1) for frame in
                     peaks.frames]  # -1 means unassigned

# Initiate lines on the first frame.
current_frame = 0
starting_frame = peaks.frames[current_frame]
tracks = [[(t, c)] for t, c in zip(starting_frame.coordinates, starting_frame.time_points)]
track_assignments[current_frame] = np.arange(starting_frame.coordinates.size)

window = 8
for idx in range(1):
    cost_matrix = []
    nexts = []
    for future_frame in np.arange(1, window):
        to_next, costs = calculate_associations(peaks.frames[idx], peaks.frames[idx + future_frame],
                                                max_distance)