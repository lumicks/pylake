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
    vector_p2_p1 = line_pt2 - line_pt1
    line_length = np.sum(vector_p2_p1 * vector_p2_p1)

    # If points coincide we have a problem
    assert line_length > 1e-8, "Line points coincide"

    distance_point_x1 = point - line_pt1
    u = np.sum(distance_point_x1 * vector_p2_p1) / line_length

    if u < 0 or u > 1:
        return np.inf

    vector_to_line = line_pt1 + u * vector_p2_p1 - point

    return np.sqrt(np.sum(vector_to_line * vector_to_line))


def minimum_distance_extrapolants(segment1, segment2):
    """Checks whether these lines can be connected.

    Checks whether the minimum distance between the extrapolant and the left side of the other segment. Then the same
    is done in the backward direction for the segment specified by other, where it evaluates the minimum distance
    between the extrapolant and the right side of this segment. The maximum of the two distances is returned.

    Parameters
    ----------
    segment1: tuple of array_like
        Tuple of two coordinates. A KymoLine end point and its linear extrapolant.
    segment2: tuple of array_like
        Tuple of two coordinates. A KymoLine start point and its linear extrapolant (towards negative time).
    """
    dist_fwd = distance_line_to_point(segment1[0], segment1[1], segment2[0])
    dist_bwd = distance_line_to_point(segment2[0], segment2[1], segment1[0])
    return dist_fwd if dist_fwd > dist_bwd else dist_bwd
