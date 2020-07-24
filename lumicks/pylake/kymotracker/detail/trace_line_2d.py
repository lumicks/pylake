import numpy as np
from .geometry_2d import is_in_2d, is_opposite, calculate_image_geometry, get_candidate_generator


def score_candidates(normal, position_difference, trial_normals, angle_weight):
    """Evaluate connection score. Connection score is made up of angular and Euclidian distance between adjacent
    pixels."""

    # We want all normals on the same side of the line, so flip those that aren't.
    flip_normal = is_opposite(trial_normals, normal)
    trial_normals[flip_normal, :] = -trial_normals[flip_normal, :]

    # Note that we have to add the relative pixel displacement on top of the subpixel displacements)
    distances = np.linalg.norm(position_difference, axis=1)
    angular_distances = np.arccos(np.dot(trial_normals, normal))
    return distances + angle_weight * angular_distances


def generate_trial_points(sign, normal, candidate_generator):
    """Generate candidate steps"""
    angle = np.arctan2(sign * normal[1], sign * normal[0])
    candidate_steps = candidate_generator(angle)
    return candidate_steps


def filter_candidates(subpixel_origin, indices, candidate_steps, masked_derivative, continuation_threshold,
                      positions, forced_direction):
    """Checks whether the candidates listed in candidate_steps are valid line candidates or not. Returns both the valid
    steps and indices."""

    candidate_indices = indices + candidate_steps
    valid = is_in_2d(candidate_indices, masked_derivative.shape)
    line_like = masked_derivative[candidate_indices[valid, 0], candidate_indices[valid, 1]] < continuation_threshold
    valid[valid] = valid[valid] & line_like

    line_steps = candidate_steps[valid, :]
    line_indices = candidate_indices[valid, :]

    subpixel_steps = line_steps + positions[line_indices[:, 0], line_indices[:, 1]] - subpixel_origin

    if forced_direction:
        if forced_direction > 0:
            direction_filter = subpixel_steps[:, 1] > 0
        else:
            direction_filter = subpixel_steps[:, 1] <= 0

        line_indices = line_indices[direction_filter, :]
        subpixel_steps = subpixel_steps[direction_filter, :]

    return line_indices, subpixel_steps


def do_step(subpixel_steps, normal, candidate_indices, candidate_normals, angle_weight):
    """Make a step in the line tracing algorithm. This function chooses a step from the candidate steps and performs it.
    It returns either a vector of new indices or None in case there is no more valid step to perform.

    Parameters
    ----------
        subpixel_steps: array_like
            List of candidate steps (in subpixel coordinates).
        normal: array_like
            Normal at the current point.
        candidate_indices: array_like
            List of candidate steps [pixel index, pixel index].
        candidate_normals: array_like
            NxMx2 array of normal vectors.
        angle_weight: array_like
            Score weighting factor.
    """
    n_options = candidate_indices.shape[0]

    if n_options == 0:
        return
    elif n_options == 1:
        # If we have only one valid option, go here for the next iteration
        return candidate_indices.flatten()
    else:
        # There are multiple valid candidates which could be an extension of the line. Compute the score for each.
        scores = score_candidates(normal, subpixel_steps, candidate_normals, angle_weight)

        # Update current point with the best scoring point
        return candidate_indices[np.argmin(scores), :]


def _traverse_line_direction(indices, masked_derivative, positions, normals, continuation_threshold, angle_weight,
                             candidate_generator, direction, force_dir):
    """Traverse a line along single direction perpendicular to the normal. While traversing, it sets visited pixels in
    the derivative image to zero to mark them as seen. Only candidates with a negative derivative are considered.

    Parameters
    ----------
    indices: array_like
        index coordinate where to start
    masked_derivative: array_like
        Image containing the second derivative masked by whether a point is a peak or not. Note that this will be
        modified in-place.
    positions: array_like
        NxMx2 array containing subpixel coordinates of the peak within this pixel.
    normals: array_like
        NxMx2 array of normal vectors. Note that normals will be flipped to maintain a fixed orientation while line
        tracing. As a result, this array will be modified in-place.
    continuation_threshold: float
        Threshold whether a line should be continued or not.
    angle_weight: float
        Factor which determines how the angle between normals needs to be weighted relative to distance.
        High values push for straighter lines. Weighting occurs according to distance + angle_weight * angle difference
    candidate_generator: callable
        Function that generate candidates pixels based on normal angle (see geometry_2d/get_candidate_generator).
    direction: float
        -1.0 or 1.0 depending on whether the algorithm should proceed forward or backward.
    force_dir: bool
        Do not allow lines to backtrack in time.
    """
    # We start walking in the desired time direction
    sign = -1.0 if direction != np.sign(normals[indices[0], indices[1]][0]) else 1.0

    forced_direction = direction if force_dir else None

    # Line strength needs to meet a threshold for continuation
    nodes = []
    while indices is not None:
        masked_derivative[indices[0], indices[1]] = 0  # Mark pixel as seen

        subpixel_origin = positions[indices[0], indices[1]]
        nodes.append(indices + subpixel_origin)

        normal = normals[indices[0], indices[1]]
        candidate_steps = generate_trial_points(sign, normal, candidate_generator)

        candidate_indices, subpixel_steps = filter_candidates(subpixel_origin, indices, candidate_steps,
                                                              masked_derivative, continuation_threshold, positions,
                                                              forced_direction)

        indices = do_step(subpixel_steps, normal, candidate_indices,
                          normals[candidate_indices[:, 0], candidate_indices[:, 1]], angle_weight)

        if indices is not None:
            # Check if normal needs to be flipped to ensure a minimal angle between the current and previous normal.
            if is_opposite(normals[indices[0], indices[1]], normal):
                normals[indices[0], indices[1]] = -normals[indices[0], indices[1]]

    return nodes


class KymoLine:
    """A line on a kymograph"""
    __slots__ = ['data']

    def __init__(self, time, coordinate):
        assert len(time) == len(coordinate)
        self.data = np.vstack((time, coordinate))

    @property
    def time(self):
        return self.data[0, :]

    @property
    def coordinate(self):
        return self.data[1, :]

    def __add__(self, other):
        return KymoLine(np.hstack((self.time, other.time)), np.hstack((self.coordinate, other.coordinate)))

    def __getitem__(self, item):
        return self.data[:, item].transpose()

    def extrapolate(self, forward, n_estimate, extrapolation_length):
        """This function linearly extrapolates a track segment towards positive time.

        Parameters
        ----------
        forward: boolean
            extrapolate forward (True) or backward in time (False)
        n_estimate: int
            Number of points to use for linear regression.
        extrapolation_length: float
            How far to extrapolate.
        """
        assert n_estimate > 1, "Too few time points to extrapolate"
        assert len(self.time) > 1, "Cannot extrapolate linearly with less than one time point"

        if forward:
            coeffs = np.polyfit(self.time[-n_estimate:], self.coordinate[-n_estimate:], 1)
            return np.array([self.time[-1] + extrapolation_length,
                             self.coordinate[-1] + coeffs[0] * extrapolation_length])
        else:
            coeffs = np.polyfit(self.time[:n_estimate], self.coordinate[:n_estimate], 1)
            return np.array([self.time[0] - extrapolation_length,
                             self.coordinate[0] - coeffs[0] * extrapolation_length])

    def __len__(self):
        return len(self.coordinate)


def traverse_line(indices, masked_derivative, positions, normals, continuation_threshold, candidate_generator,
                  angle_weight, force_dir):
    """Traverse a line in both directions."""
    indices_fwd = _traverse_line_direction(indices, masked_derivative, positions, normals, continuation_threshold,
                                           angle_weight, candidate_generator, 1, force_dir)
    indices_bwd = _traverse_line_direction(indices, masked_derivative, positions, normals, continuation_threshold,
                                           angle_weight, candidate_generator, -1, force_dir)

    indices_bwd.reverse()
    line = np.array(indices_bwd + indices_fwd)

    if len(line) > 0:
        return KymoLine(time=line[:, 1], coordinate=line[:, 0])


def detect_lines_from_geometry(masked_derivative, positions, normals, start_threshold, continuation_threshold,
                               max_lines, angle_weight, force_dir):
    """Detect lines from precomputed geometry data.

    The precomputed geometry data contains the magnitude of the second image derivative in the direction of largest
    change masked by pixels that are deemed to be a potential line center by the fact that they have their subpixel
    location in the range [-.5, .5] for both coordinates (see geometry_2d/find_subpixel_location for more information
    on how this is done). It also contains the normals to the potential line. The algorithm then traverses the line
    along the direction perpendicular to the normal vector. At each point, it will generate three candidates for a
    potential next move. If there is more than one valid candidate, a score comprised of the distance to the next
    subpixel minimum and angle between the successive normal vectors is computed. The candidate with the lowest score
    is then selected.

    As the algorithm progresses, it mutates the array masked_derivative along the way (marking seen pixels as zero,
    to avoid them from being tagged again).

    masked_derivative: array_like
        Image containing the second derivative masked by whether a point is a peak or not. Note: this array is mutated.
    positions: array_like
        NxMx2 array containing subpixel coordinates of the peak within this pixel.
    normals: array_like
        NxMx2 array of normal vectors.
    start_threshold: float
        Threshold whether a line should be initiated.
    continuation_threshold: float
        Threshold whether a line should be continued or not.
    max_lines: integer
        After how many lines should we terminate.
    angle_weight: float
        Factor which determines how the angle between normals needs to be weighted relative to distance.
        High values push for straighter lines. Weighting occurs according to distance + angle_weight * angle difference
    force_dir: bool
        Do not allow lines to backtrack in time.
    """

    def to_absolute_threshold(filtered_image, threshold):
        f = -filtered_image[filtered_image < 0]
        mn = np.min(f)
        mx = np.max(f)

        return -((mx - mn) * threshold + mn)

    # Find maximal derivative as start position
    idx = np.array(np.unravel_index(masked_derivative.argmin(), masked_derivative.shape))

    thresh = to_absolute_threshold(masked_derivative, start_threshold)
    proceed = thresh if not continuation_threshold else to_absolute_threshold(masked_derivative, continuation_threshold)

    # Generate lookup table which convert normal angle into table of points to be trialed
    candidates = get_candidate_generator()

    lines = []
    while masked_derivative[idx[0], idx[1]] < thresh and len(lines) < max_lines:
        line = traverse_line(idx, masked_derivative, positions, normals, proceed, candidates, angle_weight, force_dir)
        idx = np.array(np.unravel_index(masked_derivative.argmin(), masked_derivative.shape))
        if line:
            lines.append(line)
        else:
            break

    return lines


def detect_lines(data, line_width, start_threshold=.5, continuation_threshold=.1, max_lines=200, angle_weight=10.0,
                 force_dir=False):
    """Detect lines in an image, based on Steger at al, "An unbiased detector of curvilinear structures".

    data: np_array
        Image data.
    line_width: float
        Expected line width of the lines we are looking for.
    start_threshold: float
        Relative threshold to start tracing a candidate line. Must be between 0 and 1 (default: .5).
    continuation_threshold: float
        Relative threshold to terminate tracing a line. Must be between 0 and 1 (default: .1)
    max_lines: integer
        Maximum number of lines to detect (default: 200).
    angle_weight: float
        Factor which determines how the angle between normals needs to be weighted relative to distance.
        High values push for straighter lines. Weighting occurs according to distance + angle_weight * angle difference.
    force_dir: bool
        Do not allow lines to backtrack in time.
    """
    # See Steger et al, "An unbiased detector of curvilinear structures. IEEE Transactions on Pattern Analysis and
    # Machine Intelligence, 20(2), pp.113–125. for a motivation of this scale.
    sig_x = line_width / (2.0 * np.sqrt(3)) + .5
    sig_y = sig_x

    max_derivative, normals, positions, inside = calculate_image_geometry(data, sig_x, sig_y)

    # Mask the maximal line derivative with the point candidates
    masked_derivative = inside * max_derivative
    masked_derivative[masked_derivative > 0] = 1

    return detect_lines_from_geometry(masked_derivative, positions, normals, start_threshold, continuation_threshold,
                                      max_lines, angle_weight, force_dir)
