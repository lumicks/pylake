import numpy as np
from lumicks.pylake.kymotracker.detail.stitch import distance_line_to_point
from lumicks.pylake.kymotracker.stitching import stitch_kymo_lines
from lumicks.pylake.kymotracker.detail.calibrated_images import CalibratedKymographChannel
from lumicks.pylake.kymotracker.kymoline import KymoLine


def test_distance_line_to_point():
    assert distance_line_to_point(np.array([0, 0]), np.array([0, 1]), np.array([0, 2])) == np.inf
    assert distance_line_to_point(np.array([0, 0]), np.array([0, 2]), np.array([0, 2])) == 0.0
    assert distance_line_to_point(np.array([0, 0]), np.array([1, 1]), np.array([0, 1])) == \
           np.sqrt(0.5)
    assert distance_line_to_point(np.array([0, 0]), np.array([1, 0]), np.array([0, 1])) == 1.0


def test_stitching():
    channel = CalibratedKymographChannel("test_data", np.array([[]]), 1e9, 1)

    segment_1 = KymoLine([0, 1], [0, 1], channel)
    segment_2 = KymoLine([2, 3], [2, 3], channel)
    segment_3 = KymoLine([2, 3], [0, 0], channel)
    segment_1b = KymoLine([0, 1], [0, 0], channel)
    segment_1c = KymoLine([-1, 0, 1], [0, 0, 1], channel)

    radius = 0.05
    segment_1d = KymoLine([0.0, 1.0], [radius+.01, radius+.01], channel)

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
    assert len(stitch_kymo_lines([KymoLine([0, 1], [0, 0], channel),
                                  KymoLine([2, 2.01], [0, 0], channel)], radius, 1, 2)) == 1
    # - and | should not connect.
    assert len(stitch_kymo_lines([KymoLine([0, 1], [0, 0], channel),
                                  KymoLine([2, 2.01], [0, 1], channel)], radius, 1, 2)) == 2
