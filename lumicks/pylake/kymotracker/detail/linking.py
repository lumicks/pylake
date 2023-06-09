import enum

import numpy as np


def initialize_positions(costs):
    """Provides the initial linking matrix in coordinate form

    Parameters
    ----------
    costs : numpy.ndarray
        2D array of cost functions representing the cost of linking particle i in frame f1, to
        particle j in frame f2

    Returns
    -------
    current_to_next : numpy.ndarray
        Which particle this one connects to. The zeroth index is reserved for the dummy particle
        (which means not connected to anything).
    next_to_current : numpy.ndarray
        Which particle this one is connected from. The zeroth index is reserved for the dummy
        particle (which means not connected to anything).
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


def optimize_linking(initial_condition, cost, max_iter=100, debug=False):
    """Optimize linking of particles between frames.

    This swaps which particles are linked to minimize the cost. As long as the cost is a simple
    function that only depends on the particle indices themselves, this is guaranteed to converge
    to a global optimum.

    Parameters
    ----------
    initial_condition : tuple of numpy.ndarray
        Set of coordinate pairs with from and to coordinates. A negative index indicates unlinked.
    cost : numpy.ndarray
        2D array of cost functions representing the cost of linking particle i in frame f1, to
        particle j in frame f2
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    current_to_next : np.ndarray
        Array which contains particle links from the current frame to the next.
    next_to_current : np.ndarray
        Array which contains particle links from the next frame to the current.
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
    # 2. A particle appears at j whereas it previously had a relation to something in the previous
    #    frame (k)
    #
    #      delta_cost = cost[null, j] + cost[k, null] - cost[k, j]
    #
    # 3. A particle disappears from at i whereas it previously had a relation to something in the
    #    next frame (l)
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
        lowest_cost_change = 0
        lowest_change = None
        for i in range(1, cost.shape[0]):
            for j in range(1, cost.shape[1]):
                # Only consider adding ones that have a finite cost and don't already exist
                if cost[i, j] < np.inf and not current_to_next[i] == j:
                    # We're reassigning particles that already had a connection
                    #   i->l, k->j  =>  i->j k->l
                    k = next_to_current[j]
                    l = current_to_next[i]

                    if cost[k, l] < np.inf:
                        cost_change = cost[i, j] + cost[k, l] - cost[i, l] - cost[k, j]
                        if cost_change < lowest_cost_change:  # and np.isfinite(cost_change):
                            if debug:
                                print(
                                    f"swap (i, j): +{cost[i, j]}, (k, l): +{cost[k, l]}, (i, l): -{cost[i, l]}, (k, j): -{cost[k, j]}"
                                )
                            lowest_cost_change = cost_change
                            lowest_change = (i, j, k, l)

        # Particles disappearing
        for i in range(1, cost.shape[0]):
            #   i->l => i->0 0->l
            j = 0
            k = 0
            l = current_to_next[i]
            cost_change = cost[i, 0] + cost[0, l] - cost[i, l]
            if cost_change < lowest_cost_change:
                if debug:
                    print(
                        f"appear (i, 0): +{cost[i, 0]}, (0, l): +{cost[0, l]}, (i, l): -{cost[i, l]}"
                    )
                lowest_cost_change = cost_change
                lowest_change = (i, j, k, l)

        # Particles appearing (i becomes 0)
        for j in range(1, cost.shape[1]):
            #   k->j => 0->j k->0
            k = next_to_current[j]
            l = 0
            i = 0

            cost_change = cost[0, j] + cost[k, 0] - cost[k, j]
            if cost_change < lowest_cost_change:
                if debug:
                    print(
                        f"appear (0, j): +{cost[0, j]}, (k, 0): +{cost[k, 0]}, (k, j): -{cost[k, j]}"
                    )
                lowest_cost_change = cost_change
                lowest_change = (i, j, k, l)

        if lowest_cost_change < 0:
            # Relink particles
            #   i->l, k->j  =>  i->j k->l
            before = np.sum([cost[x + 1, y] for x, y in enumerate(current_to_next[1:])]) + cost[
                0, 0
            ] * sum(next_to_current[1:] == 0)

            if debug:
                print(current_to_next)
                print(next_to_current)

            i, j, k, l = lowest_change
            (
                next_to_current[j],
                current_to_next[i],
                next_to_current[l],
                current_to_next[k],
            ) = lowest_change

            if debug:
                print(current_to_next)
                print(next_to_current)

            after = np.sum([cost[x + 1, y] for x, y in enumerate(current_to_next[1:])]) + cost[
                0, 0
            ] * sum(next_to_current[1:] == 0)

            if debug:
                print(
                    f"Expected: {lowest_cost_change}, Realized: {after - before}, Difference (realized - expected): {after - before - lowest_cost_change}."
                )
                print(lowest_change)
            # np.testing.assert_allclose(lowest_cost_change, after - before)
        else:
            # Done!
            break

    extracted_costs = [cost[idx, target] for idx, target in enumerate(current_to_next)]

    return current_to_next, next_to_current, extracted_costs


def calculate_associations(cost_function, from_frame, to_frame, frame_difference):
    """Calculate optimal associations between particles

    Parameters
    ----------
    cost_function : callable
        Cost function that takes two frames and their distance and calculates a cost between them
        for each particle.
    from_frame, to_frame : lumicks.pylake.kymotracker.detail.peakfinding.KymoPeaks.Frame
        Particles in a frame to calculate associations for.
    frame_difference : int
        How many time frames between these two frames.
    """
    cost_matrix = cost_function(from_frame, to_frame, frame_difference)
    initial_guess = initialize_positions(cost_matrix)
    to_next, to_current, connection_cost = optimize_linking(initial_guess, cost_matrix)
    return to_next, connection_cost


def optimal_associations(cost_function, frames):
    """Calculate optimal links between frames.

    Calculates optimal links between the first frame and other frames in the provided list.

    Parameters
    ----------
    cost_function : callable
        Cost function that takes two frames and their distance and calculates a cost between them
        for each particle.
    frames : list of lumicks.pylake.kymotracker.detail.peakfinding.KymoPeaks.Frame
        Frames to calculate links for

    Returns
    -------
    link_matrix : np.ndarray
        Matrix that contains optimal particle associations between frames. This matrix has a
        row per frame and a number of particles that corresponds to the particles in the first
        frame.
    cost_matrix : np.ndarray
        Matrix with the costs of each particular linkage.
    """
    link_matrix, cost_matrix = [], []
    for frame_idx, future_frame in enumerate(frames[1:]):
        to_next, costs = calculate_associations(
            cost_function,
            frames[0],
            future_frame,
            frame_idx + 1,
        )
        link_matrix.append(to_next)
        cost_matrix.append(costs)

    return np.vstack(link_matrix), np.vstack(cost_matrix)


# Generate optimal link candidate list
def generate_links(cost_function, frames, window):
    """Generate matrix of links between frames needed for tracking

    Parameters
    ----------
    cost_function : callable
        Cost function that takes two frames and their distance and calculates a cost between them
        for each particle.
    frames : list of lumicks.pylake.kymotracker.detail.peakfinding.KymoPeaks.Frame
        Frames to calculate links for
    window : int
        Number of frames of look-ahead to compute.
    """
    link_matrices = []
    cost_matrices = []
    for idx in range(len(frames) - window):
        link_matrix, cost_matrix = optimal_associations(cost_function, frames[idx : idx + window])
        link_matrices.append(link_matrix)
        cost_matrices.append(cost_matrix)

    return link_matrices, cost_matrices


def grow_track(frames, link_matrices, seen_particles, current_frame, current_particle):
    """Grow a track from an initial frame

    Parameters
    ----------
    frames : list of lumicks.pylake.kymotracker.detail.peakfinding.KymoPeaks.Frame
        Frames to build tracks out of
    link_matrices : list
        List of link matrices. A link matrix contains optimal associations between particles
        in a frame and the frames following that frame. It contains a row per frame while
        the colummns are given by the number of particles. Note that the zeroth index is
        reserved for the "dummy" particle here. A particle which reflects a missing particle.
    seen_particles : list
        A list containing a boolean for each particle in every frame. When a particle has
        been processed, this boolean is set to True.
    current_frame : int
        Frame index to start growing from
    current_particle : int
        Particle index to start growing from.
    """
    track = []

    # If what we'd connect to has already been seen, terminate.
    if seen_particles[current_frame][current_particle]:
        return track

    while current_particle >= 0 and not seen_particles[current_frame][current_particle]:
        # Mark particle as seen to not include it twice
        seen_particles[current_frame][current_particle] = True
        track.append((current_frame, frames[current_frame].coordinates[current_particle]))

        # Grab the potential links per frame
        try:
            # Note the zeroth particle is the dummy particle here
            links = link_matrices[current_frame][:, current_particle + 1]
        except IndexError:
            # If we go beyond the end of the link matrices, end the track
            return track

        # Proceed to the next detection
        which_link = np.argmax(links > 0)

        # We advance to the next linked frame. Note that we have to add one, since
        # all the indices in the link matrix reflect future frames (e.g. the zeroth frame
        # in the link matrix is actually the frame after this one).
        current_frame += which_link + 1

        # The zeroth particle refers to the dummy particle, hence - 1 to get a real particle index
        current_particle = links[which_link] - 1

    return track


def track_greedy_optimized_association_matrix(cost_function, peaks, window):
    """Track points in a kymograph

    Parameters
    ----------
    cost_function : callable
        Cost function that takes two frames and their distance and calculates a cost between them
        for each particle.
    peaks : lumicks.pylake.kymotracker.detail.peakfinding.KymoPeaks
        Particles to track into tracks.
    window : int
        Maximum gap in a track in frames.
    """
    link_matrices_per_frame, cost_matrices_per_frame = generate_links(
        cost_function, peaks.frames, window
    )

    # Keep track of which points we've already seen
    seen_particles = [np.full(frame.coordinates.shape, False) for frame in peaks.frames]

    # Link up the detections
    current_frame = 0
    tracks = []
    for frame_idx, (link_matrix, seen) in enumerate(zip(link_matrices_per_frame, seen_particles)):
        # Link matrices provide the particle linkages. Note that particle 0 is a dummy
        # particle and should not be tracked into tracks. Let's see which particles have
        # the potential to be linked to anything, we start building tracks only from those.
        non_dummy_links = link_matrix[:, 1:]
        linkable_points = np.sum(non_dummy_links > 0, axis=0) > 0
        new_tracks = np.where(np.logical_and(np.logical_not(seen), linkable_points))[0]

        for track_idx in new_tracks:
            tracks.append(
                grow_track(
                    peaks.frames, link_matrices_per_frame, seen_particles, frame_idx, track_idx
                )
            )

    return tracks, peaks


def link_association_matrices_advancing_front(cost_function, peaks, window, debug=False):
    """Track points in a kymograph

    Parameters
    ----------
    cost_function : callable
        Cost function that takes two frames and their distance and calculates a cost between them
        for each particle.
    peaks : lumicks.pylake.kymotracker.detail.peakfinding.KymoPeaks
        Particles to track into tracks.
    window : int
        Maximum gap in a track in frames.
    """
    link_matrices_per_frame, cost_matrices_per_frame = generate_links(
        cost_function, peaks.frames, window
    )

    # Keep track of which points we've already seen
    track_assignments = [np.full(frame.coordinates.shape, -1, dtype=int) for frame in peaks.frames]

    # Initiate lines on the first frame.
    tracks = []
    for frame_idx, (link_matrix, cost_matrix) in enumerate(
        zip(link_matrices_per_frame, cost_matrices_per_frame)
    ):
        # Link matrices provide the particle correspondences. Note that particle 0 is a dummy
        # particle and should not be tracked into tracks.
        non_dummies = link_matrix[:, 1:]
        cost = cost_matrix[:, 1:]
        unassigned = np.array(
            [
                [assignments[p - 1] == -1 if p > 0 else False for p in particle_indices]
                for particle_indices, assignments in zip(
                    non_dummies, track_assignments[frame_idx + 1 : frame_idx + window]
                )
            ],
            dtype=int,
        )
        # Uncommenting this leads the algorithm to seek the next applicable point. However, what
        # this results in is tracks snaking around each-other excessively.
        # non_dummies *= unassigned

        # Find which ones we should be connecting
        first_frame = np.argmax(
            non_dummies > 0, axis=0
        )  # Find the frame in which a linkage for this particle first appears
        particle_idx = (non_dummies * unassigned)[
            first_frame, np.arange(non_dummies.shape[1])
        ]  # Find the actual particle in that frame we are linking to
        cost_values = cost[
            first_frame, np.arange(non_dummies.shape[1])
        ]  # Find the actual particle in that frame we are linking to

        # Drop dummy particles
        from_particle_idx = np.arange(particle_idx.size)[particle_idx > 0]
        first_frame = first_frame[particle_idx > 0]
        cost_values = cost_values[particle_idx > 0]
        particle_idx = particle_idx[particle_idx > 0]

        # We give particles which have a valid correspondence early priority.
        sorted_idx = np.argsort(cost_values)

        # Each linking candidates is characterized by
        # [source particle in current frame, first frame it is detected again,
        # particle index when it is detected in a future frame]
        candidates = np.vstack(
            (
                from_particle_idx[sorted_idx],
                frame_idx + first_frame[sorted_idx] + 1,
                particle_idx[sorted_idx] - 1,
            )
        )

        for from_particle, to_frame_idx, to_particle in candidates.T:
            # Check if the from particle is already part of a track. If so, we add to that track,
            # otherwise, start a new one.
            track_idx = track_assignments[frame_idx][from_particle]
            if track_idx >= 0:
                track = tracks[track_idx]
            else:
                track = [(frame_idx, peaks.frames[frame_idx].coordinates[from_particle])]
                track_idx = len(tracks)
                track_assignments[frame_idx][from_particle] = track_idx
                tracks.append(track)

            # Link up!
            if debug:
                print(
                    f"At frame {frame_idx} adding link to track {track_idx}: "
                    f"({frame_idx}, {from_particle}, {peaks.frames[frame_idx].coordinates[from_particle]}) "
                    f"-> ({to_frame_idx}, {to_particle}, {peaks.frames[to_frame_idx].coordinates[to_particle]})"
                )

            try:
                track_assignments[to_frame_idx][to_particle] = track_idx
                track.append((to_frame_idx, peaks.frames[to_frame_idx].coordinates[to_particle]))
            except IndexError:
                print(link_matrix)
                raise IndexError(":(")

    return tracks, peaks


class ParticleState(enum.IntEnum):
    unassigned = 0  # Particle has not yet been assigned
    track_start = 1  # Start of a track. It could be connected into but not from.
    track_body = 2  # This particle is at the center of a track
    track_end = 3  # End of a track. It could be connected from but not into.


def vanlierini(cost_function, peaks, window, debug=False):
    link_matrices_per_frame, cost_matrices_per_frame = generate_links(
        cost_function, peaks.frames, window
    )

    # Keep an association matrix between which point is assigned to which track
    track_assignments = [np.full(frame.coordinates.shape, -1, dtype=int) for frame in peaks.frames]

    # Keep track of which points we've already seen
    tracks = []
    to_stitch = []
    point_status = [np.zeros(frame.coordinates.shape, dtype=int) for frame in peaks.frames]

    for current_window in np.arange(2, window + 1):
        # Initiate lines on the first frame.
        for frame_idx, (link_matrix, cost_matrix) in enumerate(
            zip(link_matrices_per_frame, cost_matrices_per_frame)
        ):
            # Link matrices provide the particle correspondences. Note that particle 0 is a dummy particle and
            # should not be tracked into tracks.
            non_dummies = link_matrix[: current_window - 1, 1:]
            cost = cost_matrix[: current_window - 1, 1:]

            # Collect particle status for the points under consideration.
            particle_state = np.array(
                [
                    [status[p - 1] if p > 0 else ParticleState.track_body for p in particle_indices]
                    for particle_indices, status in zip(
                        non_dummies, point_status[frame_idx + 1 : frame_idx + current_window]
                    )
                ],
                dtype=int,
            )

            # Find which ones we should be connecting
            first_frame = np.argmax(
                non_dummies > 0, axis=0
            )  # Find the frame in which a linkage for this particle first appears
            valid_target_for_connecting_to = np.logical_or(
                particle_state == ParticleState.unassigned,
                particle_state == ParticleState.track_start,
            )

            # Find the actual particle in that frame we are linking to.
            # By multiplying with the valid target mask, we convert all connections that must not be allowed
            # into dummy particles.
            particle_idx = (non_dummies * valid_target_for_connecting_to)[
                first_frame, np.arange(non_dummies.shape[1])
            ]
            cost_values = cost[
                first_frame, np.arange(non_dummies.shape[1])
            ]  # Find the actual particle in that frame we are linking to

            # Only connect from specific points.
            valid_target_for_connecting_from = np.logical_or(
                point_status[frame_idx] == ParticleState.unassigned,
                point_status[frame_idx] == ParticleState.track_end,
            )

            # Drop dummy particles and invalid targets for starting from
            valid_connections = np.logical_and(particle_idx > 0, valid_target_for_connecting_from)

            from_particle_idx = np.arange(particle_idx.size)[valid_connections]
            first_frame = first_frame[valid_connections]
            cost_values = cost_values[valid_connections]
            particle_idx = particle_idx[valid_connections]

            # We give particles which have a valid correspondence early priority when it comes to linking.
            sorted_idx = np.argsort(cost_values)
            candidates = np.vstack(
                (
                    from_particle_idx[sorted_idx],  # source particle in current frame
                    frame_idx + first_frame[sorted_idx] + 1,  # first frame it is detected again
                    particle_idx[sorted_idx] - 1
                    # particle index when it is detected in a future frame
                )
            )

            for from_particle, to_frame_idx, to_particle in candidates.T:
                # Check if the from particle is already part of a track.
                # If so, we add to that track, otherwise, start a new one.
                track_idx = track_assignments[frame_idx][from_particle]
                if track_idx >= 0:
                    track = tracks[track_idx]

                    # It was a track end, so it now becomes a track body.
                    point_status[frame_idx][from_particle] = ParticleState.track_body
                else:
                    track = [(frame_idx, peaks.frames[frame_idx].coordinates[from_particle])]
                    track_idx = len(tracks)
                    track_assignments[frame_idx][from_particle] = track_idx
                    point_status[frame_idx][from_particle] = ParticleState.track_start
                    tracks.append(track)

                # Link up!
                if debug:
                    print(
                        f"At frame {frame_idx} adding link to track {track_idx}: "
                        f"({frame_idx}, {from_particle}, {peaks.frames[frame_idx].coordinates[from_particle]}) "
                        f"-> ({to_frame_idx}, {to_particle}, {peaks.frames[to_frame_idx].coordinates[to_particle]})"
                    )

                try:
                    if point_status[to_frame_idx][to_particle] == ParticleState.track_start:
                        # Apparently we stitched together an existing track. Don't let it be retagged.
                        point_status[to_frame_idx][to_particle] = ParticleState.track_body
                        to_stitch.append((track_idx, track_assignments[to_frame_idx][to_particle]))
                    else:
                        # Just a normal track extension
                        point_status[to_frame_idx][to_particle] = ParticleState.track_end
                        track_assignments[to_frame_idx][to_particle] = track_idx
                        track.append(
                            (to_frame_idx, peaks.frames[to_frame_idx].coordinates[to_particle])
                        )

                except IndexError:
                    print(link_matrix)
                    raise IndexError(":(")

    # Stitch from back to front
    to_stitch = np.array(to_stitch)

    change = True
    while change:
        change = False
        for to_append, from_append in to_stitch:
            if tracks[from_append]:
                change = True

            tracks[to_append] += tracks[from_append]
            tracks[from_append] = []

    tracks = [t for t in tracks if t]

    return tracks, peaks
