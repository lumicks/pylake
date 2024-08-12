import pytest

from lumicks.pylake.kymotracker.kymotrack import KymoTrack, KymoTrackGroup
from lumicks.pylake.kymotracker.detail.track_proximity import (
    BoundingBox,
    tracks_close,
    find_colocalized,
)


@pytest.mark.parametrize(
    "coordinates1, coordinates2, valid1, valid2, should_overlap",
    [
        ((2, 4, 2, 5), (1, 4, 2, 5), True, True, True),
        ((2, 4, 2, 5), (4, 7, 2, 5), True, True, True),
        ((2, 4, 2, 5), (5, 7, 2, 5), True, True, False),
        ((2, 4, 2, 5), (0, 2, 2, 5), True, True, True),
        ((2, 4, 2, 5), (0, 1, 2, 5), True, True, False),
        ((2, 4, 2, 5), (2, 4, 6, 10), True, True, False),
        ((2, 4, 2, 5), (2, 4, 0, 1), True, True, False),
        ((2, 4, 2, 5), (2, 4, 0, 10), True, True, True),
        ((2, 4, 2, 5), (0, 10, 0, 10), True, True, True),
        ((0, 0, 0, 0), (-10, 10, -10, 10), True, True, True),
        ((0, 0, 0, 0), (-10, 10, -10, 10), False, True, False),
        ((-10, 10, -10, 10), (0, 0, 0, 0), True, False, False),
    ],
)
def test_bbox_overlap_test(coordinates1, coordinates2, valid1, valid2, should_overlap):
    bbox1 = BoundingBox(*coordinates1, valid1)
    bbox2 = BoundingBox(*coordinates2, valid2)
    assert bbox1.overlaps_with(bbox2) == should_overlap
    assert bbox2.overlaps_with(bbox1) == should_overlap


@pytest.mark.parametrize(
    "time, coords, time_slack, coord_slack, ref_box, is_valid",
    [
        ([1, 2, 3, 5], [0.5, 0.8, 1.0, 2.3], 0, 0.0, (1, 5, 0.5, 2.3), True),
        ([1, 2, 3, 5], [0.5, 0.8, 1.0, 2.3], 1, 0.0, (0, 6, 0.5, 2.3), True),
        ([1, 2, 3, 5], [0.5, 0.8, 1.0, 2.3], 1, 1.0, (0, 6, -0.5, 3.3), True),
        ([0], [1], 0, 0.0, (0, 0, 1, 1), True),
        ([0], [1], 1, 0.0, (-1, 1, 1, 1), True),
        ([0], [1], 0, 1.0, (0, 0, 0, 2), True),
        ([], [], 0, 0, (0, 0, 0, 0), False),  # Empty track cannot have a valid bounding box
    ],
)
def test_bbox_construction_test(
    blank_kymo, time, coords, time_slack, coord_slack, ref_box, is_valid
):
    track = KymoTrack(time, coords, blank_kymo, "red", 0)
    bbox = BoundingBox.from_track(track, time_slack=time_slack, coord_slack=coord_slack)
    assert bbox.time_min == ref_box[0]
    assert bbox.time_max == ref_box[1]
    assert bbox.coord_min == ref_box[2]
    assert bbox.coord_max == ref_box[3]
    assert bbox._valid == is_valid


@pytest.mark.parametrize(
    "track1, track2, time_window, position_window, result",
    [
        (([1, 2, 3], [1.0, 1.0, 1.0]), ([1, 2, 3], [3.0, 3.0, 3.0]), 0, 2, True),
        (([1, 2, 3], [1.0, 1.0, 1.0]), ([1, 2, 3], [3.0, 3.0, 3.0]), 0, 1.99, False),
        (([1, 3, 5], [1.0, 1.0, 1.0]), ([2, 4, 6], [1.0, 1.0, 1.0]), 1, 1, True),
        (([1, 3, 5], [1.0, 1.0, 1.0]), ([2, 4, 6], [1.0, 1.0, 1.0]), 0, 1, False),
        (([1, 3], [1.0, 1.0]), ([2, 4, 6], [1.0, 1.0, 1.0]), 1, 1, True),
        (([1, 3], [1.0, 1.0]), ([2, 4, 6], [1.0, 1.0, 1.0]), 0, 1, False),
        (([], []), ([], []), 0, 1, False),
    ],
)
def test_colocalization_interacting(
    blank_kymo, track1, track2, time_window, position_window, result
):
    track1 = KymoTrack(*track1, blank_kymo, "red", 0)
    track2 = KymoTrack(*track2, blank_kymo, "red", 0)
    assert (
        tracks_close(track1, track2, time_window=time_window, position_window=position_window)
        == result
    )


def test_colocalization_find_colocalization(
    blank_kymo,  # time_window, position_window, result
):
    track_coords1 = [
        ([1, 2, 3], [1, 1, 1]),
        ([10, 14], [1, 1]),
    ]

    track_coords2 = [
        ([1, 2, 3], [3, 3, 3]),
        ([11, 12, 13], [2, 2, 2]),  # Will only show up when colocalized
    ]

    tracks1 = KymoTrackGroup([KymoTrack(*tc, blank_kymo, "red", 0) for tc in track_coords1])
    tracks2 = KymoTrackGroup([KymoTrack(*tc, blank_kymo, "red", 0) for tc in track_coords2])

    # assert (
    find_colocalized(
        tracks1, tracks2, time_window=1, position_window=1
    )  # , time_window=time_window, position_window=position_window)
    #    == result
    # )
