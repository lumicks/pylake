import numpy as np
from copy import deepcopy


def distance_line_to_point(line_pt1, line_pt2, point):
    """Find smallest distance between line segment and point. Returns np.inf if shortest distance is outside line segment.

    P = P1 + u * (P2 - P1)

    At the closest point, we are tangent to the line, therefore
        dot(point - P, P2 - P1) = 0.

    Inserting the line equation for P and solving for u gives:
        u = ((point_x - Px1)(Px2 - Px1) + (point_y - Py1)(Py2 - Py1)) / norm(P2 - P1)
    """
    distance_p2_p1 = line_pt2 - line_pt1
    line_length = np.sum(distance_p2_p1 * distance_p2_p1)

    # If points coincide we have a problem
    assert line_length > 1e-8, "Line points coincide"

    distance_point_x1 = point - line_pt1
    u = np.sum(distance_point_x1 * distance_p2_p1) / line_length

    if u < 0 or u > 1:
        return np.inf

    vector_to_line = line_pt1 + u * (line_pt2 - line_pt1) - point

    return np.sqrt(np.sum(vector_to_line * vector_to_line))


def stitch_kymo_lines(lines, radius, max_extension, n_points):
    """This function stitches KymoLines together.

    The stitching algorithm works as follows. The following steps are repeated for every line:

    Step 1. We regress a linear function to the last `n_points` of the line.
    Step 2. We extrapolate by a user-set time `max_extension`. Ideally, this should be chosen as the largest gap size.
    Step 3. Based on this extrapolated line, we compute the minimum distance between this line and all leftmost points of all other segments.
    Step 4. If this closest point is within the line segment *and* the minimum distance below a user defined radius, the point is acceptable.
    Step 5. For all acceptable points we compute the distance to the current line segment. We take that one as segment with minimal distance to connect to.

    Parameters
    ----------
    lines: list of KymoLine
        Lines to stitch.
    radius: float
        Any valid extension should fall within this number of pixels of the other.
    max_extension: float
        Duration to extrapolate. Setting this higher will extrapolate further, leading to larger spaces being crossed.
    n_points: int
        Number of points to use for the linear regression.
    """
    stitched_lines = deepcopy(lines)

    origin_idx = 0
    while origin_idx < len(stitched_lines):
        # Perform linear regression on the last few points
        origin_line = stitched_lines[origin_idx]

        # Compute distance to all other lines
        distances = np.array([np.linalg.norm(origin_line[-1] - target_line[0], 2) for target_line in stitched_lines])

        perpendicular_distance = np.array(
            [origin_line.connects_linear(line, max_extension, n_points) for line in stitched_lines])

        # Make sure our origin line isn't a candidate
        distances[origin_idx] = np.inf

        # Filter unacceptable distances
        not_acceptable = perpendicular_distance > radius
        distances[not_acceptable] = np.inf

        # Determine nearest point
        min_dist_idx = np.argmin(distances)

        # Determine vertical distance

        # Check if distance lower than linking threshold
        if distances[min_dist_idx] < np.inf:
            stitched_lines[origin_idx] = stitched_lines[origin_idx] + stitched_lines[min_dist_idx]
            del stitched_lines[min_dist_idx]
            if origin_idx > min_dist_idx:
                origin_idx -= 1
        else:
            origin_idx += 1

    return stitched_lines
