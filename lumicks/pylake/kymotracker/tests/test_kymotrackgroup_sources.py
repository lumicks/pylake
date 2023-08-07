import re

import pytest

from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.kymotracker.kymotrack import *


@pytest.fixture(scope="module")
def kymos():
    image = np.random.poisson(10, size=(10, 10, 3))

    def make_kymo(line_time_seconds, pixel_size_um):
        return _kymo_from_array(
            image,
            "rgb",
            start=np.int64(20e9),
            line_time_seconds=line_time_seconds,
            pixel_size_um=pixel_size_um,
            name="big & long",
        )

    return (
        make_kymo(10e-4, 0.05),
        make_kymo(10e-4, 0.1),
        make_kymo(10e-3, 0.05),
        make_kymo(10e-3, 0.1),
    )


@pytest.fixture(scope="module")
def coordinates():
    time_indices = ([1, 2, 3], [4, 6, 7], [1, 2, 3], [4, 6, 7])
    position_indices = ([4, 5, 6], [1, 7, 7], [1, 2, 3], [1, 2, 3])
    return time_indices, position_indices


def check_attributes(tracks, expected_line_times, expected_pixel_sizes, expected_calibration_unit):
    assert set([kymo.line_time_seconds for kymo in tracks._kymos]) == set(expected_line_times)
    assert set([kymo.pixelsize_um[0] for kymo in tracks._kymos]) == set(expected_pixel_sizes)
    assert set([kymo._calibration.unit for kymo in tracks._kymos]) == set(expected_calibration_unit)


def test_empty_constructor():
    tracks = KymoTrackGroup([])
    assert len(tracks) == 0
    assert tracks._kymos == ()
    check_attributes(tracks, [], [], [])

    tracks = KymoTrackGroup([]) + tracks
    assert len(tracks) == 0
    assert tracks._kymos == ()
    check_attributes(tracks, [], [], [])

    with pytest.raises(
        RuntimeError,
        match=re.escape("No channel associated with this empty group (no tracks available)"),
    ):
        tracks._channel

    with pytest.raises(
        RuntimeError,
        match=re.escape("No kymo associated with this empty group (no tracks available)"),
    ):
        tracks.plot()


def test_constructor(kymos, coordinates):
    kymo = kymos[0]
    time_indices, position_indices = coordinates
    raw_tracks = [
        KymoTrack(t, p, kymo, "green", kymo.line_time_seconds)
        for t, p in zip(time_indices, position_indices)
    ]
    raw_tracks_red = [
        KymoTrack(t, p, kymo, "red", kymo.line_time_seconds)
        for t, p in zip(time_indices, position_indices)
    ]

    # construct from single source
    tracks = KymoTrackGroup(raw_tracks)
    assert len(tracks) == 4
    assert tracks._kymos == (kymo,)
    check_attributes(tracks, [10e-4], [0.05], ["um"])

    # construct with duplicate track
    with pytest.raises(
        ValueError,
        match=re.escape("Some tracks appear multiple times. The provided tracks must be unique."),
    ):
        tracks = KymoTrackGroup([*raw_tracks, raw_tracks[0]])

    # construct from single source, different channels
    with pytest.raises(
        ValueError,
        match=re.escape("All tracks must be from the same color channel."),
    ):
        tracks = KymoTrackGroup([*raw_tracks[:2], *raw_tracks_red[2:]])


def test_extend_single_source(kymos, coordinates):
    kymo = kymos[0]
    time_indices, position_indices = coordinates
    raw_tracks = [
        KymoTrack(t, p, kymo, "green", kymo.line_time_seconds)
        for t, p in zip(time_indices, position_indices)
    ]
    raw_tracks_red = [
        KymoTrack(t, p, kymo, "red", kymo.line_time_seconds)
        for t, p in zip(time_indices, position_indices)
    ]

    tracks1 = KymoTrackGroup(raw_tracks[:2])
    tracks2 = KymoTrackGroup(raw_tracks[2:])

    # add track
    tracks = tracks1 + raw_tracks[2]
    assert len(tracks) == 3
    assert tracks._kymos == (kymo,)
    check_attributes(tracks, [10e-4], [0.05], ["um"])

    # add group
    tracks = tracks1 + tracks2
    assert len(tracks) == 4
    assert tracks._kymos == (kymo,)
    check_attributes(tracks, [10e-4], [0.05], ["um"])

    # extend with single source, different channels
    with pytest.raises(
        ValueError,
        match=re.escape("All tracks must be from the same color channel."),
    ):
        tracks1 + KymoTrackGroup(raw_tracks_red)


def test_extend_empty(kymos, coordinates):
    kymo = kymos[0]
    time_indices, position_indices = coordinates
    raw_tracks = [
        KymoTrack(t, p, kymo, "green", kymo.line_time_seconds)
        for t, p in zip(time_indices, position_indices)
    ]

    empty = KymoTrackGroup([])
    tracks2 = KymoTrackGroup(raw_tracks)

    # can't add a group to a track
    with pytest.raises(
        AttributeError, match="'KymoTrackGroup' object has no attribute '_localization'"
    ):
        tracks = tracks2[0] + empty

    # add track
    tracks = empty + tracks2[0]
    assert len(tracks) == 1
    assert tracks._kymos == (kymo,)
    check_attributes(tracks, [10e-4], [0.05], ["um"])

    # add group
    tracks = empty + tracks2
    assert len(tracks) == 4
    assert tracks._kymos == (kymo,)
    check_attributes(tracks, [10e-4], [0.05], ["um"])

    tracks = tracks2 + empty
    assert len(tracks) == 4
    assert tracks._kymos == (kymo,)
    check_attributes(tracks, [10e-4], [0.05], ["um"])


def test_different_sources_same_attributes(kymos, coordinates):
    kymo1 = kymos[0]
    kymo2 = copy(kymos[0])
    assert id(kymo1) != id(kymo2)

    time_indices, position_indices = coordinates
    tracks1 = KymoTrackGroup(
        [
            KymoTrack(t, p, kymo1, "green", kymo1.line_time_seconds)
            for t, p in zip(time_indices, position_indices)
        ]
    )
    tracks2 = KymoTrackGroup(
        [
            KymoTrack(t, p, kymo2, "green", kymo2.line_time_seconds)
            for t, p in zip(time_indices, position_indices)
        ]
    )

    tracks = tracks1[:2] + tracks2[2:]
    assert len(tracks) == 4
    assert tracks._kymos == (kymo1, kymo2)
    check_attributes(tracks, [10e-4], [0.05], ["um"])


def test_different_sources_different_attributes(kymos, coordinates):
    time_indices, position_indices = coordinates

    def make_tracks(kymo):
        return KymoTrackGroup(
            [
                KymoTrack(t, p, kymo, "green", kymo.line_time_seconds)
                for t, p in zip(time_indices, position_indices)
            ]
        )

    # different line times
    tracks1 = make_tracks(kymos[0])
    tracks2 = make_tracks(kymos[2])

    kymo_bp = kymos[0].calibrate_to_kbp(48)
    tracks2_bp = make_tracks(kymo_bp)

    tracks = tracks1 + tracks2
    assert len(tracks) == 8
    assert tracks._kymos == (kymos[0], kymos[2])
    check_attributes(tracks, [10e-4, 10e-3], [0.05], ["um"])

    # different pixel sizes
    tracks1 = make_tracks(kymos[0])
    tracks2 = make_tracks(kymos[1])

    tracks = tracks1 + tracks2
    assert len(tracks) == 8
    assert tracks._kymos == (kymos[0], kymos[1])
    check_attributes(tracks, [10e-4], [0.05, 0.1], ["um"])

    # different line times and pixel sizes
    tracks1 = make_tracks(kymos[0])
    tracks2 = make_tracks(kymos[-1])

    tracks = tracks1 + tracks2
    assert len(tracks) == 8
    assert tracks._kymos == (kymos[0], kymos[-1])
    check_attributes(tracks, [10e-4, 10e-3], [0.05, 0.1], ["um"])

    # extend with different calibrations
    with pytest.raises(
        ValueError,
        match=r"All tracks must be calibrated in the same units, got {.+}\.",
    ):
        tracks1 + tracks2_bp


def test_tracks_by_kymo(kymos, coordinates):
    time_indices, position_indices = coordinates

    def make_tracks(kymo):
        return KymoTrackGroup(
            [
                KymoTrack(t, p, kymo, "green", kymo.line_time_seconds)
                for t, p in zip(time_indices, position_indices)
            ]
        )

    tracks = [make_tracks(k) for k in (kymos[0], kymos[-1], kymos[0])]
    merged_group = tracks[0] + tracks[1] + tracks[2]
    tracks_from_group, indices = merged_group._tracks_by_kymo()
    grouped_by_kymo = [tracks[0] + tracks[2], tracks[1]]

    for tracks_raw, tracks_group in zip(grouped_by_kymo, tracks_from_group):
        assert len(tracks_group) == len(tracks_raw)
        for track_group, track_raw in zip(tracks_group, tracks_raw):
            id(track_group) == id(track_raw)

    assert indices == [[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7]]
