import numpy as np


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

    return np.sum(vector_to_line * vector_to_line)
