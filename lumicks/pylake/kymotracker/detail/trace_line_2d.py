import enum
import numpy as np
from .geometry_2d import is_in_2d, is_opposite, calculate_image_geometry, get_candidate_generator
from .scoring_functions import build_score_matrix


class KymoCode(enum.IntEnum):
    seen = 0  # This pixel was already processed


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
        masked_derivative[indices[0], indices[1]] = KymoCode.seen  # Mark pixel as seen

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
    __slots__ = ['time', 'coordinate']

    def __init__(self, time, coordinate):
        self.time = list(time)
        self.coordinate = list(coordinate)

    def append(self, time, coordinate):
        """Append time, coordinate pair to the KymoLine"""
        self.time.append(time)
        self.coordinate.append(coordinate)

    def with_offset(self, time_offset, coordinate_offset):
        """Returns an offset version of the KymoLine"""
        return KymoLine([time + time_offset for time in self.time],
                        [coordinate + coordinate_offset for coordinate in self.coordinate])

    def __add__(self, other):
        """Concatenate two KymoLines"""
        return KymoLine(self.time + other.time, self.coordinate + other.coordinate)

    def __getitem__(self, item):
        return np.squeeze(np.array(np.vstack((self.time[item], self.coordinate[item]))).transpose())

    def in_rect(self, rect):
        """Check whether any point of this KymoLine falls in the rect given in rect.

        Parameter
        ---------
        rect : Tuple[Tuple[float, float], Tuple[float, float]]
            Only perform tracking over a subset of the image. Pixel coordinates should be given as:
            ((min_time, min_coord), (max_time, max_coord)).
        """
        time = np.array(self.time)
        coordinate = np.array(self.coordinate)
        time_match = np.logical_and(time < rect[1][0], time >= rect[0][0])
        coord_match = np.logical_and(coordinate < rect[1][1], coordinate >= rect[0][1])
        return np.any(np.logical_and(time_match, coord_match))

    def sample_from_image(self, data, num_pixels, reduce=np.sum):
        """Sample from image using coordinates from this KymoLine.

        This function samples data from the image given in data based on the points in this KymoLine. It samples
        from [time, position - num_pixels : position + num_pixels + 1] and then applies the function sum.

        Parameters
        ----------
        data : array_like
            Image to sample from using this KymoLine.
        num_pixels : int
            Number of pixels in either direction to include in the sample
        reduce : callable
            Function evaluated on the sample. (Default: np.sum which produces sum of photon counts).
        """
        y_size = data.shape[1]

        # Time and coordinates are being cast to an integer since we use them to index into a data array.
        return [reduce(data[max(int(c) - num_pixels, 0):min(int(c) + num_pixels + 1, y_size), int(t)])
                for t, c in zip(self.time, self.coordinate)]

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

        time = np.array(self.time)
        coordinate = np.array(self.coordinate)

        if forward:
            coeffs = np.polyfit(time[-n_estimate:], coordinate[-n_estimate:], 1)
            return np.array([time[-1] + extrapolation_length,
                             coordinate[-1] + coeffs[0] * extrapolation_length])
        else:
            coeffs = np.polyfit(time[:n_estimate], coordinate[:n_estimate], 1)
            return np.array([time[0] - extrapolation_length,
                             coordinate[0] - coeffs[0] * extrapolation_length])

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

    As the algorithm progresses, it mutates the array masked_derivative along the way (marking seen pixels to avoid
    them from being tagged again).

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

    thresh = to_absolute_threshold(masked_derivative, start_threshold)
    proceed = thresh if not continuation_threshold else to_absolute_threshold(masked_derivative, continuation_threshold)

    # Generate lookup table which convert normal angle into table of points to be trialed
    candidates = get_candidate_generator()

    lines = []
    for flat_idx in np.argsort(masked_derivative.flatten()):
        idx = np.unravel_index(flat_idx, masked_derivative.shape)

        if masked_derivative[idx[0], idx[1]] == KymoCode.seen:
            continue

        if masked_derivative[idx[0], idx[1]] >= thresh or len(lines) > max_lines:
            break

        # Traverse the line. Note that traverse_line modifies the masked_derivative image by marking some as seen.
        line = traverse_line(idx, masked_derivative, positions, normals, proceed, candidates, angle_weight, force_dir)
        if line:
            lines.append(line)

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


def append_next_point(line, frame, score_fun):
    """Scores potential peak points and selects the most optimal one. If an acceptable point is found, the function
    returns True, adds the point to the line and marks it as assigned in the frame.

    Parameters
    ----------
    line : pylake.kymotracker.trace_line_2d.KymoLine
    frame : KymoPeaks.kymotracker.peakfinding.KymoPeaks.Frame
    score_fun : callable
        Function which takes a list of N lines and M points (in the form of candidate_times and candidate_coordinates)
        and computes a score for each line, point combination resulting in an N by M matrix of scores. Returns -np.inf
        for points that should not be considered viable candidates.
    """
    if np.any(frame.unassigned):
        candidate_idx, = np.where(frame.unassigned)
        candidate_times = frame.time_points[candidate_idx]
        candidate_coordinates = frame.coordinates[candidate_idx]

        score_matrix = score_fun([line], candidate_times, candidate_coordinates)
        selected = np.argmax(score_matrix)

        if not np.isinf(score_matrix[selected]):
            line.append(candidate_times[selected], candidate_coordinates[selected])
            frame.unassigned[candidate_idx[selected]] = False
            return True

    return False


def extend_line(line, peaks, window, score_fun):
    """Extend a line. This extension terminates when it can't find an extension in the next "window" frames.

    Parameters
    ----------
    line : pylake.kymotracker.trace_line_2d.KymoLine
        Contains the line currently being traced.
    peaks : KymoPeaks.kymotracker.peakfinding.KymoPeaks
        Contains all the peak data.
    window : int
        How many frames do we allow a peak to disappear?
    score_fun : callable
        Function which takes a list of N lines and M points (in the form of candidate_times and candidate_coordinates)
        and computes a score for each line, point combination resulting in an N by M matrix of scores. Returns -np.inf
        for points that should not be considered viable candidates.
    """
    frames_without_peak = 0
    starting_frame = int(line.time[-1]) + 1
    for current_frame in peaks.frames[starting_frame:]:
        found_next_point = append_next_point(line, current_frame, score_fun)
        frames_without_peak = 0 if found_next_point else frames_without_peak + 1
        if frames_without_peak >= window:
            break


def points_to_line_segments(peaks, prediction_model, window=10, sigma_cutoff=2):
    """Starts from a list of coordinates and attempts to string them together.

        For each frame:
        - All points that haven't been assigned yet are considered line starts. We start a line at point with the
          highest signal.
        - For each line:
          - We check whether we can extend the line to the next frame by connecting it to the most likely next point.
          - If we cannot find a point that is sufficiently likely, we iteratively try the next N window frames.
          - If we've exhausted the maximum number of window frames to look ahead, we terminate the line.
        - When there are no more line starts, go to the next frame.

    Which point to connect to is determined by considering a prediction model. This prediction model returns a mu and
    sigma that describes a Gaussian curve which reflects the probability of finding a particle in a certain area
    on a future frame. In addition to a maximum window (maximum number of frames that a particle is expected to be able
    to disappear), there is also a sigma_cutoff parameter. This parameter controls the width of the cone in which
    particles may be accepted. Setting this value to two (meaning two sigma), means you'd accept the most optimal point
    falling within two sigma or 95.45% of the mean of the prediction. The lower this value, the fewer points you'll
    accept and the narrower you expect lines to be.

    Parameters
    ----------
    peaks: KymoPeaks.kymotracker.peakfinding.KymoPeaks
        peaks identified as potential lines.
    window: int
        How many frames can a particle disappear before we assume it isn't the same line.
    sigma_cutoff: float
        sigma cutoff points for the classification on whether it could belong to the same line.
    prediction_model : callable
        Function which takes a line and produces a mu and sigma for a list of coordinates.
    """
    peaks.reset_assignment()

    def score_matrix(line_list, time, coord):
        return build_score_matrix(line_list, time, coord, prediction_model, sigma_cutoff=sigma_cutoff).flatten()

    lines = []
    for frame in peaks.frames:
        # Give precedence to lines with higher peak amplitudes
        for starting_point in np.argsort(-frame.peak_amplitudes * frame.unassigned):
            if frame.unassigned[starting_point]:
                line = KymoLine([frame.time_points[starting_point]], [frame.coordinates[starting_point]])
                frame.unassigned[starting_point] = False

                extend_line(line, peaks, window, score_matrix)
                lines.append(line)

    return lines
