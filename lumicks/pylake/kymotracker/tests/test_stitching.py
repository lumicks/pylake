import numpy as np
from lumicks.pylake.kymotracker.detail.stitch import distance_line_to_point
from lumicks.pylake.kymotracker.stitching import stitch_kymo_lines
from lumicks.pylake.kymotracker.kymotrack import KymoTrack


def test_distance_line_to_point():
    assert distance_line_to_point(np.array([0, 0]), np.array([0, 1]), np.array([0, 2])) == np.inf
    assert distance_line_to_point(np.array([0, 0]), np.array([0, 2]), np.array([0, 2])) == 0.0
    assert distance_line_to_point(np.array([0, 0]), np.array([1, 1]), np.array([0, 1])) == np.sqrt(
        0.5
    )
    assert distance_line_to_point(np.array([0, 0]), np.array([1, 0]), np.array([0, 1])) == 1.0


def test_stitching(blank_kymo):

    segment_1 = KymoTrack([0, 1], [0, 1], blank_kymo, "red")
    segment_2 = KymoTrack([2, 3], [2, 3], blank_kymo, "red")
    segment_3 = KymoTrack([2, 3], [0, 0], blank_kymo, "red")
    segment_1b = KymoTrack([0, 1], [0, 0], blank_kymo, "red")
    segment_1c = KymoTrack([-1, 0, 1], [0, 0, 1], blank_kymo, "red")

    radius = 0.05
    segment_1d = KymoTrack([0.0, 1.0], [radius + 0.01, radius + 0.01], blank_kymo, "red")

    # Out of stitch range (maximum extension = 1)
    assert len(stitch_kymo_lines([segment_1, segment_3, segment_2], radius, 1, 2)) == 3

    # Out of stitch radius
    assert len(stitch_kymo_lines([segment_1d, segment_3, segment_2], radius, 2, 2)) == 3

    stitched = stitch_kymo_lines([segment_1, segment_3, segment_2], radius, 2, 2)
    assert len(stitched) == 2
    np.testing.assert_allclose(stitched[0].coordinate_idx, [0, 1, 2, 3])
    np.testing.assert_allclose(stitched[1].coordinate_idx, [0, 0])

    stitched = stitch_kymo_lines([segment_1b, segment_3, segment_2], radius, 2, 2)
    np.testing.assert_allclose(stitched[0].coordinate_idx, [0, 0, 0, 0])
    np.testing.assert_allclose(stitched[0].time_idx, [0, 1, 2, 3])
    np.testing.assert_allclose(stitched[1].coordinate_idx, [2, 3])

    # Check whether only the last two points are used (meaning we extrapolate [0, 0], [1, 1])
    stitched = stitch_kymo_lines([segment_1c, segment_3, segment_2], radius, 2, 2)
    np.testing.assert_allclose(stitched[0].coordinate_idx, [0, 0, 1, 2, 3])
    np.testing.assert_allclose(stitched[0].time_idx, [-1, 0, 1, 2, 3])

    # When using all three points, we shouldn't stitch
    assert len(stitch_kymo_lines([segment_1c, segment_3, segment_2], radius, 2, 3)) == 3

    # Check whether the alignment has to work in both directions
    # - and - should connect
    track1, track2 = KymoTrack([0, 1], [0, 0], blank_kymo, "red"), KymoTrack([2, 2.01], [0, 0], blank_kymo, "red")
    assert len(stitch_kymo_lines([track1, track2], radius, 1, 2)) == 1

    # - and | should not connect.
    track1, track2 = KymoTrack([0, 1], [0, 0], blank_kymo, "red"), KymoTrack([2, 2.01], [0, 1], blank_kymo, "red")
    assert len(stitch_kymo_lines([track1, track2], radius, 1, 2)) == 2
