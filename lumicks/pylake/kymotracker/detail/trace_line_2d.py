import numpy as np
from .geometry_2d import is_in_2d, is_opposite


def _traverse_line_direction(indices, masked_derivative, positions, normals, continuation_threshold, angle_weight,
                             candidates, sign=1.0, keep_first=True):
    """Traverse a line along single direction perpendicular to the normal.

    indices: array_like
        index coordinate where to start
    masked_derivative: array_like
        Image containing the second derivative masked by whether a point is a peak or not.
    positions: array_like
        NxMx2 array containing subpixel coordinates of the peak within this pixel.
    normals: array_like
        NxMx2 array of normal vectors.
    continuation_threshold: float
        Threshold whether a line should be continued or not.
    angle_weight: float
        How strongly does the angle between the normals factor into the penalty.
    candidates: callable
        Function that generate candidates pixels based on normal angle (see geometry_2d/get_candidate_generator).
    sign: float
        -1.0 or 1.0 depending on whether the algorithm should proceed forward or backward.
    keep_first: boolean
        Do not remove starting point
    """

    # Line strength needs to meet a threshold for continuation
    nodes = []
    while masked_derivative[indices[0], indices[1]] < continuation_threshold:
        if not keep_first:
            # Mark pixel as seen
            masked_derivative[indices[0], indices[1]] = 0
        else:
            keep_first = False

        # Determine normal at origin
        normal = normals[indices[0], indices[1]]
        subpixel_origin = positions[indices[0], indices[1]]

        nodes.append(indices + subpixel_origin)

        # Generate trial points given our current normal angle
        angle = np.arctan2(sign * normal[1], sign * normal[0])
        candidate_steps = candidates(angle)
        candidate_indices = indices + candidate_steps

        # Reject candidates that aren't line points
        valid = is_in_2d(candidate_indices, masked_derivative.shape)
        valid[valid] = valid[valid] & (masked_derivative[candidate_indices[valid, 0], candidate_indices[valid, 1]] < 0)
        n_options = np.sum(valid)

        # We ran out of options. Terminate the line.
        if n_options == 0:
            break

        # If we have only one valid option, go here for the next iteration
        if n_options == 1:
            indices = candidate_indices[valid, :].flatten()
        else:
            # There is ambiguity to be resolved in the search direction
            candidate_steps = candidate_steps[valid, :]
            candidate_indices = candidate_indices[valid, :]

            # We want all normals on the same side of the line, so flip those that aren't.
            trial_normals = normals[candidate_indices[:, 0], candidate_indices[:, 1]]
            flip_normal = is_opposite(trial_normals, normal)
            trial_normals[flip_normal, :] = -trial_normals[flip_normal, :]

            # Score is made up of angular and Euclidian distance. Pixel with minimum score wins.
            # Note that we have to add the relative pixel displacement on top of the subpixel displacements)
            trial_positions = candidate_steps + positions[candidate_indices[:, 0], candidate_indices[:, 1]]
            distances = np.linalg.norm(trial_positions - subpixel_origin, axis=1)
            angular_distances = np.arccos(np.dot(trial_normals, normal))
            scores = distances + angle_weight * angular_distances

            # Update current point with the best scoring point
            indices = candidate_indices[np.argmin(scores), :]

        # Check if normal needs to be flipped to ensure a minimal angle between the current and previous normal vector.
        if is_opposite(normals[indices[0], indices[1]], normal):
            normals[indices[0], indices[1]] = -normals[indices[0], indices[1]]

    return nodes


def traverse_line(indices, masked_derivative, positions, normals, continuation_threshold, angle_weight=1.0):
    indices_fwd = _traverse_line_direction(indices, masked_derivative, positions, normals, continuation_threshold,
                                           angle_weight, 1, True)
    indices_bwd = _traverse_line_direction(indices, masked_derivative, positions, normals, continuation_threshold,
                                           angle_weight, -1, False)
    indices_fwd.reverse()
    return np.array(indices_fwd + indices_bwd)