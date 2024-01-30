import re
from copy import copy

import pytest
import matplotlib.pyplot as plt

from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.kymotracker.kymotrack import *
from lumicks.pylake.kymotracker.kymotracker import filter_tracks, _to_half_kernel_size
from lumicks.pylake.kymotracker.detail.localization_models import *

from ...tests.data.mock_confocal import generate_kymo
from ...population.tests.data.generate_exponential_data import ExponentialParameters, make_dataset


def test_kymo_track(blank_kymo, blank_kymo_track_args):
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([2, 3, 4]), *blank_kymo_track_args)
    assert k1._minimum_observable_duration == blank_kymo.line_time_seconds
    np.testing.assert_allclose(k1[1], [2, 3])
    np.testing.assert_allclose(k1[-1], [3, 4])
    np.testing.assert_allclose(k1[0:2], [[1, 2], [2, 3]])
    np.testing.assert_allclose(k1[0:2][:, 1], [2, 3])

    k2 = KymoTrack(np.array([4, 5, 6]), np.array([5, 6, 7]), *blank_kymo_track_args)
    np.testing.assert_allclose((k1 + k2)[:], [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    np.testing.assert_allclose(k1.extrapolate(True, 3, 2.0), [5, 6])

    with pytest.raises(
        RuntimeError, match="Cannot extrapolate linearly with fewer than two timepoints"
    ):
        KymoTrack([1], [1], *blank_kymo_track_args).extrapolate(True, 5, 2.0)

    with pytest.raises(
        ValueError, match="Cannot extrapolate linearly with fewer than two timepoints"
    ):
        KymoTrack([1, 2, 3], [1, 2, 3], *blank_kymo_track_args).extrapolate(True, 1, 2.0)

    k1 = KymoTrack([1, 2, 3], [1, 2, 3], *blank_kymo_track_args)
    k2 = k1.with_offset(2, 2)
    assert id(k2) != id(k1)
    assert k2._minimum_observable_duration == blank_kymo.line_time_seconds

    np.testing.assert_allclose(k2.coordinate_idx, [3, 4, 5])
    assert k1._kymo == blank_kymo
    assert k2._kymo == blank_kymo
    assert k1._channel == "red"
    assert k2._channel == "red"
    assert str(k1) == "KymoTrack(N=3)"


@pytest.mark.parametrize(
    "time_idx, ref_value", [[[0, 1, 2], 2], [[0, 2], 2], [[1, 3], 2], [[1, 4, 5], 4]]
)
def test_kymotrack_duration(blank_kymo, time_idx, ref_value):
    np.testing.assert_allclose(
        KymoTrack(time_idx, time_idx, blank_kymo, "red", 0).duration,
        blank_kymo.line_time_seconds * ref_value,
    )


def test_kymotrack_selection(blank_kymo, blank_kymo_track_args):
    t = np.array([4, 5, 6])
    y = np.array([7, 7, 7])
    assert not KymoTrack(t, y, *blank_kymo_track_args).in_rect(((4, 6), (6, 7)))
    assert KymoTrack(t, y, *blank_kymo_track_args).in_rect(((4, 6), (6, 8)))
    assert not KymoTrack(t, y, *blank_kymo_track_args).in_rect(((3, 6), (4, 8)))
    assert KymoTrack(t, y, *blank_kymo_track_args).in_rect(((3, 6), (5, 8)))

    assert KymoTrack([2], [6], *blank_kymo_track_args).in_rect(((2, 5), (3, 8)))
    assert KymoTrack([2], [5], *blank_kymo_track_args).in_rect(((2, 5), (3, 8)))
    assert not KymoTrack([4], [6], *blank_kymo_track_args).in_rect(((2, 5), (3, 8)))
    assert not KymoTrack([1], [6], *blank_kymo_track_args).in_rect(((2, 5), (3, 8)))
    assert not KymoTrack([2], [4], *blank_kymo_track_args).in_rect(((2, 5), (3, 8)))
    assert not KymoTrack([2], [8], *blank_kymo_track_args).in_rect(((2, 5), (3, 8)))


def test_kymotrack_selection_non_unit_calibration():
    kymo = generate_kymo(
        "",
        np.ones((1, 3)),
        pixel_size_nm=5000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0,
    )

    time, pos = np.array([4, 5, 6]), np.array([7, 7, 7])
    track = KymoTrack(time, pos, kymo, "red", 0)
    assert not track.in_rect(((4, 6 * 5), (6, 7 * 5)))
    assert track.in_rect(((4, 6 * 5), (6, 8 * 5)))


@pytest.mark.parametrize(
    "rect, remaining_lines, fully_in_rect",
    [
        ([[5, 3], [6, 4]], [True, True, False], False),
        ([[6, 3], [5, 4]], [True, True, False], False),
        ([[6, 5], [5, 3]], [True, True, False], False),
        ([[1, 1], [4, 2]], [False, True, True], True),
        ([[1, 1], [4, 2]], [False, True, True], False),
        ([[1, 1], [3, 2]], [True, True, True], True),  # Not fully inside
        ([[1, 1], [3, 2]], [False, True, True], False),
        ([[0, 0], [5, 5]], [False, False, False], False),
        ([[15, 3], [16, 4]], [True, True, True], False),
    ],
)
def test_kymotracks_removal(blank_kymo, rect, remaining_lines, fully_in_rect):
    """Tests removal of KymoTracks within a particular rectangle.

    Note that the rectangle is given as (min time, min coord) to (max time, max coord)"""
    shared_args = [blank_kymo, "red", blank_kymo.line_time_seconds]
    k0 = KymoTrack(np.array([1, 2, 3]), np.array([1, 1, 1]), *shared_args)
    k1 = KymoTrack(np.array([2, 3, 4]), np.array([2, 2, 2]), *shared_args)
    k2 = KymoTrack(np.array([3, 4, 5]), np.array([3, 3, 3]), *shared_args)
    tracks = [k0, k1, k2]

    def verify(rect, resulting_tracks):
        k = KymoTrackGroup(tracks)
        k.remove_tracks_in_rect(rect, fully_in_rect)
        assert len(k._src) == len(resulting_tracks)
        assert all([l1 == l2 for l1, l2 in zip(k, resulting_tracks)])

    verify(rect, [track for track, should_remain in zip(tracks, remaining_lines) if should_remain])


def test_kymotrack_group_getitem(blank_kymo):
    shared_args = [blank_kymo, "red", blank_kymo.line_time_seconds]
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([2, 3, 4]), *shared_args)
    k2 = KymoTrack(np.array([2, 3, 4]), np.array([3, 4, 5]), *shared_args)
    k3 = KymoTrack(np.array([3, 4, 5]), np.array([4, 5, 6]), *shared_args)
    k4 = KymoTrack(np.array([4, 5, 6]), np.array([5, 6, 7]), *shared_args)

    tracks = KymoTrackGroup([k1, k2, k3, k4])
    assert [k for k in tracks] == [k1, k2, k3, k4]
    assert len(tracks) == 4
    assert tracks[0] == k1
    assert tracks[1] == k2
    assert tracks[-1] == k4
    assert len(tracks[0:2]) == 2
    assert tracks[0:2][0] == k1
    assert tracks[0:2][1] == k2
    assert tracks[1:3][0] == k2
    assert tracks[1:3][1] == k3
    assert tracks[2:-1][0] == k3
    assert len(tracks[2:-1]) == 1
    assert tracks[2:][0] == k3
    assert tracks[2:][1] == k4

    with pytest.raises(IndexError):
        tracks[0:2][2]

    with pytest.raises(NotImplementedError):
        tracks[1] = 4


@pytest.mark.parametrize(
    "remove,remaining",
    [
        ([1], [True, False, True]),
        ([0, 1], [False, False, True]),
        ([2, 1], [True, False, False]),
    ],
)
def test_kymotrackgroup_remove(blank_kymo, remove, remaining):
    shared_args = [blank_kymo, "red", blank_kymo.line_time_seconds]
    src_tracks = [
        KymoTrack(np.array([1, 2, 3]), np.array([2, 3, 4]), *shared_args),
        KymoTrack(np.array([2, 3, 4]), np.array([3, 4, 5]), *shared_args),
        KymoTrack(np.array([3, 4, 5]), np.array([4, 5, 6]), *shared_args),
    ]

    tracks = KymoTrackGroup(src_tracks)
    for track in remove:
        tracks.remove(src_tracks[track])
    for track, should_be_present in zip(src_tracks, remaining):
        if remaining:
            assert track in tracks
        else:
            assert track not in tracks


def test_kymotrack_group_unique_construction(blank_kymo):
    shared_args = [blank_kymo, "red", blank_kymo.line_time_seconds]
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([2, 3, 4]), *shared_args)
    k2 = KymoTrack(np.array([2, 3, 4]), np.array([3, 4, 5]), *shared_args)
    with pytest.raises(
        ValueError, match="Some tracks appear multiple times. The provided tracks must be unique."
    ):
        KymoTrackGroup([k1, k2, k2])

    KymoTrackGroup([])  # An empty one should be ok


def test_kymotrackgroup_lookup(blank_kymo):
    shared_args = [blank_kymo, "red", blank_kymo.line_time_seconds]
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([2, 3, 4]), *shared_args)
    k2 = KymoTrack(np.array([2, 3, 4]), np.array([3, 4, 5]), *shared_args)
    k3 = KymoTrack(np.array([3, 4, 5]), np.array([4, 5, 6]), *shared_args)

    tracks = KymoTrackGroup([k1, k2, k3])
    assert tracks._get_track_by_id(id(k1)) is k1
    assert tracks._get_track_by_id(id(k2)) is k2
    tracks.remove(k2)
    assert tracks._get_track_by_id(id(k2)) is None
    assert tracks._get_track_by_id(id(k3)) is k3
    assert tracks._get_track_by_id(-1) is None

    tracks = KymoTrackGroup([])
    assert tracks._get_track_by_id(id(k1)) is None


def test_kymotrack_group_indexing(blank_kymo):
    shared_args = [blank_kymo, "red", blank_kymo.line_time_seconds]
    k0 = KymoTrack(np.array([1, 2, 3, 4]), np.array([2, 3, 4, 5]), *shared_args)
    k1 = KymoTrack(np.array([2, 3, 4, 5, 6]), np.array([3, 4, 5, 6, 7]), *shared_args)
    k2 = KymoTrack(np.array([3, 4, 5]), np.array([4, 5, 6]), *shared_args)
    group = KymoTrackGroup([k0, k1, k2])

    def verify_group(group, ref_list):
        assert len(group) == len(ref_list)
        for t1, t2 in zip(group, ref_list):
            assert t1 is t2

    # Boolean array indexing
    verify_group(group[[len(t) > 3 for t in group]], [k0, k1])
    verify_group(group[[len(t) < 4 for t in group]], [k2])

    # Regular array indexing
    verify_group(group[np.asarray([0, 2])], [k0, k2])
    verify_group(group[[0, 2]], [k0, k2])
    verify_group(group[[-2]], [k1])


def test_kymotrack_group_extend(blank_kymo):
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([2, 3, 4]), blank_kymo, "red", 0)
    k2 = KymoTrack(np.array([2, 3, 4]), np.array([3, 4, 5]), blank_kymo, "red", 0)
    k3 = KymoTrack(np.array([3, 4, 5]), np.array([4, 5, 6]), blank_kymo, "red", 0)
    k4 = KymoTrack(np.array([4, 5, 6]), np.array([5, 6, 7]), blank_kymo, "red", 0)

    tracks = KymoTrackGroup([k1, k2])
    tracks.extend(KymoTrackGroup([k3, k4]))
    assert [k for k in tracks] == [k1, k2, k3, k4]

    tracks = KymoTrackGroup([k1, k2, k3])
    tracks.extend(k4)
    assert [k for k in tracks] == [k1, k2, k3, k4]

    tracks = KymoTrackGroup([k1, k2, k3])

    duplicate_error = (
        "Cannot extend this KymoTrackGroup with a KymoTrack that is already part of the group"
    )

    for extension, exception, error in (
        (5, TypeError, "You can only extend a KymoTrackGroup with a KymoTrackGroup or KymoTrack"),
        (KymoTrackGroup([k3, k4]), ValueError, duplicate_error),
        (k3, ValueError, duplicate_error),
    ):
        with pytest.raises(exception, match=error):
            tracks.extend(extension)

        # Validate that we did not modify the list when it failed
        assert [k for k in tracks] == [k1, k2, k3]


def test_kymotrack_group(blank_kymo):
    def validate_same(kymoline_group, ref_list, source_items, ref_kymo):
        assert [k for k in kymoline_group] == ref_list
        assert id(kymoline_group) not in (id(s) for s in source_items)
        if ref_kymo:
            assert id(kymoline_group._kymos[0]) == id(ref_kymo)

    k1 = KymoTrack(np.array([1, 2, 3]), np.array([2, 3, 4]), blank_kymo, "red", 0)
    k2 = KymoTrack(np.array([2, 3, 4]), np.array([3, 4, 5]), blank_kymo, "red", 0)
    k3 = KymoTrack(np.array([3, 4, 5]), np.array([4, 5, 6]), blank_kymo, "red", 0)
    k4 = KymoTrack(np.array([4, 5, 6]), np.array([5, 6, 7]), blank_kymo, "red", 0)

    tracks1 = KymoTrackGroup([k1, k2])
    tracks2 = KymoTrackGroup([k3, k4])
    empty_tracks = KymoTrackGroup([])

    validate_same(tracks1 + tracks2, [k1, k2, k3, k4], {tracks1, tracks2}, k1._kymo)
    validate_same(tracks1 + k4, [k1, k2, k4], {tracks1, k4}, k1._kymo)
    validate_same(tracks1 + empty_tracks, [k1, k2], {tracks1, empty_tracks}, k1._kymo)
    validate_same(empty_tracks + tracks2, [k3, k4], {empty_tracks, tracks2}, k3._kymo)
    validate_same(empty_tracks + empty_tracks, [], {empty_tracks}, None)

    with pytest.raises(
        TypeError, match="You can only extend a KymoTrackGroup with a KymoTrackGroup or KymoTrack"
    ):
        tracks1 + 5


def test_kymotrack_merge():
    image = np.random.randint(0, 20, size=(10, 10, 3))
    kwargs = dict(line_time_seconds=10e-3, start=np.int64(20e9), pixel_size_um=0.05, name="test")
    kymo = _kymo_from_array(image, "rgb", **kwargs)

    time_idx = ([1, 2, 3, 4, 5], [6, 7, 8], [6, 7, 8], [1, 2, 3])
    pos_idx = ([1, 1, 1, 3, 3], [4, 4, 4], [9, 9, 9], [10, 10, 10])

    make_tracks = lambda: KymoTrackGroup(
        [KymoTrack(t, p, kymo, "green", 0) for t, p in zip(time_idx, pos_idx)]
    )

    # connect first two
    tracks = make_tracks()
    tracks._merge_tracks(tracks[0], 2, tracks[1], 1)
    assert len(tracks) == 3
    np.testing.assert_equal(tracks[0].time_idx, [1, 2, 3, 7, 8])
    np.testing.assert_almost_equal(tracks[0].coordinate_idx, [1, 1, 1, 4, 4])
    np.testing.assert_equal(tracks[1].time_idx, [6, 7, 8])
    np.testing.assert_almost_equal(tracks[1].coordinate_idx, [9, 9, 9])

    # connect last two
    tracks = make_tracks()
    tracks._merge_tracks(tracks[1], 1, tracks[2], 2)
    assert len(tracks) == 3
    np.testing.assert_equal(tracks[0].time_idx, [1, 2, 3, 4, 5])
    np.testing.assert_almost_equal(tracks[0].coordinate_idx, [1, 1, 1, 3, 3])
    np.testing.assert_equal(tracks[1].time_idx, [6, 7, 8])
    np.testing.assert_almost_equal(tracks[1].coordinate_idx, [4, 4, 9])

    # connect first and last
    tracks = make_tracks()
    tracks._merge_tracks(tracks[0], 3, tracks[2], 1)
    assert len(tracks) == 3
    np.testing.assert_equal(tracks[0].time_idx, [1, 2, 3, 4, 7, 8])
    np.testing.assert_almost_equal(tracks[0].coordinate_idx, [1, 1, 1, 3, 9, 9])
    np.testing.assert_equal(tracks[1].time_idx, [6, 7, 8])
    np.testing.assert_almost_equal(tracks[1].coordinate_idx, [4, 4, 4])

    # can't connect tracks from two groups
    tracks2 = KymoTrackGroup([KymoTrack([1, 2, 3], [4, 5, 6], kymo, "green", 0)])
    with pytest.raises(
        RuntimeError, match="Both tracks need to be part of this group to be merged"
    ):
        tracks._merge_tracks(tracks[0], 2, tracks2[0], 0)

    # first node must be before second node, else switch order
    tracks = make_tracks()
    tracks._merge_tracks(tracks[0], 3, tracks[-1], 1)
    np.testing.assert_equal(tracks[0].time_idx, [6, 7, 8])
    np.testing.assert_almost_equal(tracks[0].coordinate_idx, [4, 4, 4])
    np.testing.assert_equal(tracks[1].time_idx, [6, 7, 8])
    np.testing.assert_almost_equal(tracks[1].coordinate_idx, [9, 9, 9])
    np.testing.assert_equal(tracks[2].time_idx, [1, 2, 4, 5])
    np.testing.assert_almost_equal(tracks[2].coordinate_idx, [10, 10, 3, 3])

    tracks = make_tracks()
    with pytest.raises(ValueError, match="Cannot connect two points with the same time index."):
        tracks._merge_tracks(tracks[0], 1, tracks[-1], 1)


@pytest.mark.parametrize(
    "time_scale, position_scale",
    [
        (1, 1),
        (2, 1),
        (1, 2),
        (1, 0.5),
    ],
)
def test_msd_api(time_scale, position_scale):
    # Tests whether the calibrations from the image are picked up correctly through the API
    time_idx = np.arange(25) * 4
    position_idx = np.arange(25.0) * 2.0

    # What we do is we apply the scaling to the indices used in the KymoTrack construction. If we
    # define the calibration as exactly the opposite of this, we should get no change.
    kymo = generate_kymo(
        "",
        np.ones((1, 3)),
        pixel_size_nm=1000 / position_scale,
        start=np.int64(20e9),
        dt=np.int64(1e9 / time_scale),
        samples_per_pixel=1,
        line_padding=0,
    )
    k = KymoTrack(time_scale * time_idx, position_scale * position_idx, kymo, "red", 0)

    lags, msd = k.msd()
    np.testing.assert_allclose(lags, time_idx[1:])
    np.testing.assert_allclose(msd, position_idx[1:] ** 2)

    lags, msd = k.msd(max_lag=4)
    np.testing.assert_allclose(lags, time_idx[1:5])
    np.testing.assert_allclose(msd, position_idx[1:5] ** 2)


@pytest.mark.parametrize(
    "time_idx, coordinate, pixel_size, time_step, max_lag, diffusion_const",
    [
        (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]), 2, 0.5e9, 50, 4.53333333),
        (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]), 3, 1.0e9, 50, 2.26666667),
        (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]), 5, 1.0e9, 2, 3.33333333),
    ],
)
def test_diffusion_msd(time_idx, coordinate, pixel_size, time_step, max_lag, diffusion_const):
    """Tests whether the calibrations from the image are picked up correctly through the API.
    The actual tests of the diffusion estimation can be found in test_msd."""

    # What we do is we apply the scaling to the indices used in the KymoTrack construction. If we
    # define the calibration as exactly the opposite of this, we should get no change.
    kymo = generate_kymo(
        "",
        np.ones((1, 3)),
        pixel_size_nm=1000 * pixel_size,
        start=np.int64(20e9),
        dt=np.int64(time_step),
        samples_per_pixel=1,
        line_padding=0,
    )
    k = KymoTrack(time_idx, coordinate / pixel_size, kymo, "red", 0)

    np.testing.assert_allclose(k.estimate_diffusion("ols", max_lag=max_lag).value, diffusion_const)


@pytest.mark.parametrize("calibration_coeff", [0.5, 2.0])
def test_diffusion_units(blank_kymo, calibration_coeff):
    kymotrack, kymotrack_kbp = [
        KymoTrack(
            np.arange(1, 6),
            np.array([-1.0, 1.0, -1.0, -3.0, -5.0]),
            kymo,
            "red",
            kymo.line_time_seconds,
        )
        for kymo in (blank_kymo, blank_kymo.calibrate_to_kbp(calibration_coeff))
    ]

    ref_constant = 3.33333333333
    diffusion_estimate = kymotrack_kbp.estimate_diffusion("ols", max_lag=2)
    np.testing.assert_allclose(diffusion_estimate.value, ref_constant * calibration_coeff**2)
    assert diffusion_estimate.unit == "kbp^2 / s"

    diffusion_estimate = kymotrack.estimate_diffusion("ols", max_lag=2)
    np.testing.assert_allclose(diffusion_estimate.value, ref_constant)
    assert diffusion_estimate.unit == "um^2 / s"
    assert diffusion_estimate._unit_label == "μm²/s"


@pytest.mark.parametrize(
    "time_idx, coordinate, pixel_size, time_step, max_lag, diffusion_const",
    [
        (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]), 2, 0.5e9, 50, 5.609315381569765),
        (
            np.arange(1, 6),
            np.array([-1.0, 1.0, -1.0, -3.0, -5.0]),
            3,
            1.0e9,
            50,
            2.8046576907848824,
        ),
        (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]), 5, 1.0e9, 2, 3.33333333333332),
    ],
)
def test_diffusion_gls(time_idx, coordinate, pixel_size, time_step, max_lag, diffusion_const):
    """Tests whether the calibrations from the image are picked up correctly through the API.
    The actual tests of the diffusion estimation can be found in test_msd."""

    # What we do is we apply the scaling to the indices used in the KymoTrack construction. If we
    # define the calibration as exactly the opposite of this, we should get no change.
    kymo = generate_kymo(
        "",
        np.ones((1, 3)),
        pixel_size_nm=1000 * pixel_size,
        start=np.int64(20e9),
        dt=np.int64(time_step),
        samples_per_pixel=1,
        line_padding=0,
    )
    k = KymoTrack(time_idx, coordinate / pixel_size, kymo, "red", kymo.line_time_seconds)

    np.testing.assert_allclose(k.estimate_diffusion("gls", max_lag=max_lag).value, diffusion_const)


def test_invalid_method(blank_kymo, blank_kymo_track_args):
    k = KymoTrack(np.arange(5), np.arange(5), *blank_kymo_track_args)
    with pytest.raises(ValueError, match="Invalid method selected"):
        k.estimate_diffusion(max_lag=5, method="BAD")


def test_lag_default(blank_kymo, blank_kymo_track_args):
    """Checks whether the max_lag argument is correctly set by default"""
    k = KymoTrack(np.arange(5), np.arange(5), *blank_kymo_track_args)
    assert k.estimate_diffusion(method="gls").num_lags == 5  # Uses all lags by default
    assert k.estimate_diffusion(method="ols").num_lags == 2  # Should estimate lags itself (2)
    assert k.estimate_diffusion(method="ols", max_lag=4).num_lags == 4
    assert k.estimate_diffusion(method="gls", max_lag=4).num_lags == 4


@pytest.mark.parametrize(
    "max_lag, x_data, y_data",
    [
        (None, [2, 4], (3 * np.arange(1, 3)) ** 2),
        (100, [2, 4, 6, 8], (3 * np.arange(1, 5)) ** 2),
        (3, [2, 4, 6], (3 * np.arange(1, 4)) ** 2),
    ],
)
def test_kymotrack_msd_plot(max_lag, x_data, y_data):
    # See whether the plot spins up
    kymo = generate_kymo(
        "",
        np.ones((1, 3)),
        pixel_size_nm=3000,
        start=np.int64(20e9),
        dt=np.int64(2e9),
        samples_per_pixel=1,
        line_padding=0,
    )

    plt.figure()
    k = KymoTrack(np.arange(1, 6), np.arange(1, 6), kymo, "red", kymo.line_time_seconds)
    k.plot_msd(max_lag=max_lag)
    np.testing.assert_allclose(plt.gca().lines[0].get_xdata(), x_data)
    np.testing.assert_allclose(plt.gca().lines[0].get_ydata(), y_data)


def test_binding_histograms():
    kymo = generate_kymo(
        "",
        np.zeros((10, 10)),
        pixel_size_nm=1000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0,
    )

    k1 = KymoTrack(np.array([1, 2, 3]), np.array([2.5, 3.5, 4.5]), kymo, "red", 0)
    k2 = KymoTrack(np.array([2, 3, 4]), np.array([3.5, 4.5, 5.5]), kymo, "red", 0)
    k3 = KymoTrack(np.array([3, 4, 5]), np.array([4.5, 5.5, 6.5]), kymo, "red", 0)
    k4 = KymoTrack(np.array([4, 5, 6]), np.array([5.5, 6.5, 7.5]), kymo, "red", 0)

    tracks = KymoTrackGroup([k1, k2, k3, k4])

    # Counting only the first position of each track with the default number of bins
    counts, edges = tracks._histogram_binding_events("binding")
    np.testing.assert_equal(counts, [0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(edges, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Counting all points of each track with the default number of bins
    counts, edges = tracks._histogram_binding_events("all")
    np.testing.assert_equal(counts, [0, 0, 1, 2, 3, 3, 2, 1, 0, 0])
    np.testing.assert_allclose(edges, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Counting only the first position of each track with custom bin edges
    counts, edges = tracks._histogram_binding_events("binding", bins=[2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_equal(counts, [1, 1, 1, 1, 0, 0])
    np.testing.assert_allclose(edges, [2, 3, 4, 5, 6, 7, 8])

    # Counting all points of each track with custom bin edges
    counts, edges = tracks._histogram_binding_events("all", bins=[2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_equal(counts, [1, 2, 3, 3, 2, 1])
    np.testing.assert_allclose(edges, [2, 3, 4, 5, 6, 7, 8])


def test_empty_binding_histogram():
    with pytest.raises(RuntimeError, match="No tracks available for analysis"):
        KymoTrackGroup([]).plot_binding_histogram("binding")


def test_kymotrackgroup_tracks_in_frame(blank_kymo):
    tracks = KymoTrackGroup([])
    assert len(tracks._tracks_in_frame(0)) == 0

    tracks = KymoTrackGroup(
        [
            KymoTrack(np.array([0, 3, 5]), np.array([1.0, 3.0, 3.0]), blank_kymo, "red", 0),
            KymoTrack(np.array([3, 4, 5]), np.array([1.0, 3.0, 3.0]), blank_kymo, "red", 0),
        ]
    )

    # For each of the KymoTrack frames we get a list of track indices and indices that correspond to
    # the index within the track of where we intersect with this frame of the kymograph.
    reference_values = (
        (0, [(0, 0)]),
        (1, []),
        (2, []),
        (3, [(0, 1), (1, 0)]),
        (4, [(1, 1)]),
        (5, [(0, 2), (1, 2)]),
    )
    for kymo_frame_idx, reference_data in reference_values:
        tracks_in_frame = tracks._tracks_in_frame(kymo_frame_idx)
        assert len(tracks_in_frame) == len(reference_data)
        for track_data, reference_track_data in zip(tracks_in_frame, reference_data):
            np.testing.assert_equal(track_data, reference_track_data)


def test_kymotrackgroup_copy(blank_kymo):
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([1, 1, 1]), blank_kymo, "red", 0)
    k2 = KymoTrack(np.array([6, 7, 8]), np.array([2, 2, 2]), blank_kymo, "red", 0)
    group = KymoTrackGroup([k1, k2])
    assert id(group._src) != id(copy(group)._src)


def test_kymotrack_split(blank_kymo):
    k1 = KymoTrack(
        np.array([1, 2, 3]),
        np.array([1, 3, 4]),
        blank_kymo,
        "red",
        2 * blank_kymo.line_time_seconds,
    )
    k2, k3 = k1._split(1)
    np.testing.assert_allclose(k2.position, [1])
    np.testing.assert_allclose(k3.position, [3, 4])
    np.testing.assert_allclose(k2._minimum_observable_duration, 2 * blank_kymo.line_time_seconds)
    np.testing.assert_allclose(k3._minimum_observable_duration, 2 * blank_kymo.line_time_seconds)

    k2, k3 = k1._split(2)
    np.testing.assert_allclose(k2.position, [1, 3])
    np.testing.assert_allclose(k3.position, [4])
    np.testing.assert_allclose(k2._minimum_observable_duration, 2 * blank_kymo.line_time_seconds)
    np.testing.assert_allclose(k3._minimum_observable_duration, 2 * blank_kymo.line_time_seconds)

    # Test corner cases
    for split_point in [-1, 0, 3]:
        with pytest.raises(ValueError, match="Invalid split point"):
            k1._split(split_point)


def test_kymotrackgroup_split(blank_kymo):
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([1, 3, 4]), blank_kymo, "red", 0)
    k2 = KymoTrack(np.array([1, 2, 3]), np.array([8, 9, 11]), blank_kymo, "red", 0)

    group = KymoTrackGroup([k1, k2])
    assert len(group) == 2
    group._split_track(k1, 1, min_length=1)
    assert len(group) == 3
    np.testing.assert_allclose(group[-2].position, [1])
    np.testing.assert_allclose(group[-1].position, [3, 4])
    group._split_track(k2, 2, min_length=1)
    assert len(group) == 4
    np.testing.assert_allclose(group[-2].position, [8, 9])
    np.testing.assert_allclose(group[-1].position, [11])


@pytest.mark.parametrize(
    "split, filter, filtered",
    [
        (5, 5, 0),
        (4, 5, 1),
        (6, 5, 1),
        (4, 4, 0),
    ],
)
def test_kymotrackgroup_split_filter(blank_kymo, split, filter, filtered):
    k1 = KymoTrack(np.arange(10), np.arange(10), blank_kymo, "red", 0)
    group = KymoTrackGroup([k1])

    group._split_track(k1, split, min_length=filter)
    assert len(group) == 2 - filtered


def test_kymotrack_concat_gaussians(blank_kymo):
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([1, 1, 1]), blank_kymo, "red", 0)
    k1g = KymoTrack(
        np.array([1, 2, 3]),
        GaussianLocalizationModel(
            np.full(3, 2), np.full(3, 7), np.full(3, 0.5), np.ones(3), np.full(3, False)
        ),
        blank_kymo,
        "red",
        0,
    )
    k2 = KymoTrack(
        np.array([6, 7, 8]),
        GaussianLocalizationModel(
            np.full(3, 2), np.full(3, 7), np.full(3, 0.5), np.ones(3), np.full(3, False)
        ),
        blank_kymo,
        "red",
        0,
    )
    group = KymoTrackGroup([k1, k2])

    assert isinstance(k1._localization, LocalizationModel)
    assert isinstance(k2._localization, GaussianLocalizationModel)

    # test merging gaussian and non-gaussian localized lines clears gaussian parameters
    group._merge_tracks(k1, 2, k2, 2)
    assert len(group) == 1
    assert isinstance(group[0]._localization, LocalizationModel)

    # test whether merging two gaussian localized lines preserves the localization
    group = KymoTrackGroup([k1g, k2])
    group._merge_tracks(k1g, 2, k2, 2)
    assert len(group) == 1
    assert isinstance(group[0]._localization, GaussianLocalizationModel)


@pytest.mark.parametrize("num_pixels, pixelsize_um", [(10, 0.1), (15, 0.2)])
def test_kymotrack_flip(num_pixels, pixelsize_um):
    kymo = _kymo_from_array(
        np.ones((num_pixels, 10)), "r", line_time_seconds=1e-2, pixel_size_um=pixelsize_um
    )
    coords = np.arange(1.0, 4.0)

    track = KymoTrack([1, 2, 3], coords / pixelsize_um, kymo, "red", kymo.line_time_seconds)
    flipped_kymo = kymo.flip()
    flipped_track = track._flip(flipped_kymo)
    assert id(track) != id(flipped_track)
    assert flipped_track._minimum_observable_duration == track._minimum_observable_duration

    center_to_center_um = pixelsize_um * (num_pixels - 1)
    np.testing.assert_allclose(flipped_track.position, center_to_center_um - coords)
    np.testing.assert_equal(flipped_track.time_idx, track.time_idx)
    assert id(flipped_track._kymo) == id(flipped_kymo)
    assert flipped_track._channel == track._channel

    np.testing.assert_allclose(flipped_track._flip(kymo).position, track.position)


def test_kymotrackgroup_flip():
    kymo = _kymo_from_array(np.ones((5, 10)), "r", line_time_seconds=1e-2, pixel_size_um=1.0)
    coords = np.arange(1.0, 4.0)

    tracks = KymoTrackGroup(
        [KymoTrack([1, 2, 3], coords, kymo, "red", kymo.line_time_seconds) for _ in range(3)]
    )
    flipped_tracks = tracks._flip()
    assert all([id(t._kymo) == id(flipped_tracks[0]._kymo) for t in flipped_tracks])
    for track, flipped_track in zip(tracks, flipped_tracks):
        np.testing.assert_allclose(track._flip(kymo.flip()).position, flipped_track.position)

    tracks2 = tracks + KymoTrackGroup(
        [KymoTrack([], [], copy(kymo), "red", kymo.line_time_seconds)]
    )
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "Flipping is not supported. This group contains tracks from 2 source kymographs."
        ),
    ):
        tracks2._flip()

    with pytest.raises(
        RuntimeError,
        match=re.escape("No kymo associated with this empty group (no tracks available)"),
    ):
        KymoTrackGroup([])._flip()


def test_binding_profile_histogram():
    kymo = generate_kymo(
        "",
        np.ones((10, 10)),
        pixel_size_nm=1000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0,
    )

    k1 = KymoTrack(np.array([1, 2, 3]), np.array([2, 3, 4]), kymo, "red", 0)
    k2 = KymoTrack(np.array([2, 3, 4]), np.array([3, 4, 5]), kymo, "red", 0)
    k3 = KymoTrack(np.array([3, 4, 5]), np.array([4, 5, 6]), kymo, "red", 0)
    k4 = KymoTrack(np.array([4, 5, 6]), np.array([5, 6, 7]), kymo, "red", 0)

    tracks = KymoTrackGroup([k1, k2, k3, k4])

    # full kymo
    x, y_data = tracks._histogram_binding_profile(3, 0.2, 4)
    y_ref = [
        [3.86752108e-022, 1.00000000e000, 4.14559001e-073, 3.96110737e-266],
        [8.32495048e-087, 2.32557804e-002, 1.55038536e-002, 5.54996698e-087],
        [1.98055368e-266, 2.07279501e-073, 5.00000000e-001, 2.77988974e-049],
    ]

    np.testing.assert_allclose(x, np.linspace(0, 10, 4))
    for j, (y, ref) in enumerate(zip(y_data, y_ref)):
        np.testing.assert_allclose(y, ref, err_msg=f"failed on item {j}")

    # ROI
    x, y_data = tracks._histogram_binding_profile(3, 0.2, 4, roi=[[15, 3], [40, 6]])
    y_ref = [
        [1.24221001e-06, 6.42912623e-23, 4.62111561e-50, 4.61295977e-88],
        [6.66662526e-01, 2.48442002e-06, 1.28582525e-22, 9.24223122e-50],
        [3.72663003e-06, 9.99997516e-01, 1.00000000e00, 6.66667495e-01],
    ]

    np.testing.assert_allclose(x, np.linspace(3, 6, 4))
    for j, (y, ref) in enumerate(zip(y_data, y_ref)):
        np.testing.assert_allclose(y, ref, err_msg=f"failed on item {j}")

    # test empty bin than frames
    x, densities = tracks._histogram_binding_profile(10, 0.2, 4)
    for j, d in enumerate(densities):
        if j in (0, 7, 8, 9):
            np.testing.assert_equal(d, 0)

    # test no spatial bins
    with pytest.raises(ValueError, match="Number of spatial bins must be >= 2."):
        tracks._histogram_binding_profile(11, 0.2, 0)

    # test more bins than frames
    with pytest.raises(ValueError, match="Number of time bins must be <= number of frames."):
        tracks._histogram_binding_profile(11, 0.2, 4)

    # no bins requested
    with pytest.raises(ValueError, match="Number of time bins must be > 0."):
        tracks._histogram_binding_profile(0, 0.2, 4)

    # disallowed for multiple source kymos
    combined_tracks = tracks + KymoTrackGroup(
        [KymoTrack(np.array([7, 8, 9]), np.array([1, 1, 1]), copy(kymo), "red", 0)]
    )
    with pytest.raises(
        NotImplementedError,
        match=(
            r"Binding profile is not supported. This group contains tracks from 2 source kymographs."
        ),
    ):
        combined_tracks._histogram_binding_profile(n_time_bins=1, bandwidth=1, n_position_points=10)

    with pytest.raises(
        RuntimeError,
        match=re.escape("No kymo associated with this empty group (no tracks available)"),
    ):
        KymoTrackGroup([])._histogram_binding_profile(3, 0.2, 4)


@pytest.mark.parametrize(
    "discrete, exclude_ambiguous_dwells, ref_value",
    [
        (False, True, 1.002547),
        (False, False, 1.25710457),
        (True, True, 1.46272938),
        (True, False, 1.73359969),
    ],
)
def test_fit_binding_times(
    blank_kymo, blank_kymo_track_args, discrete, exclude_ambiguous_dwells, ref_value
):
    k1 = KymoTrack(np.array([0, 1, 2]), np.zeros(3), *blank_kymo_track_args)
    k2 = KymoTrack(np.array([2, 3, 4, 5, 6]), np.zeros(5), *blank_kymo_track_args)
    k3 = KymoTrack(np.array([3, 4, 5]), np.zeros(3), *blank_kymo_track_args)
    k4 = KymoTrack(np.array([8, 9]), np.zeros(2), *blank_kymo_track_args)

    tracks = KymoTrackGroup([k1, k2, k3, k4])

    dwells = tracks.fit_binding_times(
        1,
        observed_minimum=True,
        exclude_ambiguous_dwells=exclude_ambiguous_dwells,
        discrete_model=discrete,
    )
    np.testing.assert_allclose(dwells.lifetimes, [ref_value])


def test_fit_binding_times_nonzero(blank_kymo, blank_kymo_track_args):
    k1 = KymoTrack(np.array([2]), np.zeros(3), *blank_kymo_track_args)
    k2, k3, k4, k5 = (
        KymoTrack(np.array([2, 3, 4, 5, 6]), np.zeros(5), *blank_kymo_track_args) for _ in range(4)
    )
    tracks = KymoTrackGroup([k1, k2, k3, k4, k5])

    with pytest.warns(
        RuntimeWarning,
        match=r"Some dwell times are zero. A dwell time of zero indicates that some of the tracks "
        r"were only observed in a single frame. For these samples it is not possible to "
        r"actually determine a dwell time. Therefore these samples are dropped from the "
        r"analysis. If you wish to not see this warning, filter the tracks with "
        r"`lk.filter_tracks` with a minimum length of 2 samples.",
    ):
        dwelltime_model = tracks.fit_binding_times(
            n_components=1, observed_minimum=True, discrete_model=False
        )
        np.testing.assert_equal(dwelltime_model.dwelltimes, [4, 4, 4, 4])
        np.testing.assert_equal(dwelltime_model._observation_limits[0], 4)
        np.testing.assert_allclose(dwelltime_model.lifetimes[0], [0.4])


def test_fit_binding_times_empty():
    with pytest.raises(RuntimeError, match="No tracks available for analysis"):
        KymoTrackGroup([]).fit_binding_times(1)


def test_fit_binding_times_warning(blank_kymo_track_args):
    ktg = KymoTrackGroup(
        [KymoTrack(np.array([1, 2, 3]), np.array([1, 2, 3]), *blank_kymo_track_args)]
    )
    with pytest.warns(UserWarning, match="use `observed_minimum=False`"):
        ktg.fit_binding_times(1, discrete_model=False)

    with pytest.warns(UserWarning, match="pass the parameter `discrete_model=True`"):
        ktg.fit_binding_times(1, observed_minimum=False)


@pytest.mark.parametrize(
    "observed_minimum, ref_minima",
    [
        (False, [10e-4, 10e-4, 10e-3, 10e-3]),
        (True, [0.002, 0.002, 0.02, 0.02]),
    ],
)
def test_multi_kymo_dwell(observed_minimum, ref_minima):
    kymos = [
        _kymo_from_array(np.zeros((10, 10, 3)), "rgb", line_time_seconds=time, pixel_size_um=size)
        for time, size in ((10e-4, 0.05), (10e-3, 0.1))
    ]

    line_times = [k.line_time_seconds for k in kymos]
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([1, 2, 3]), kymos[0], "red", line_times[0])
    k2 = KymoTrack(np.array([2, 3, 4, 5]), np.array([2, 3, 4, 5]), kymos[0], "red", line_times[0])
    k3 = KymoTrack(np.array([3, 4, 5]), np.array([3, 4, 5]), kymos[1], "red", line_times[1])
    k4 = KymoTrack(np.array([4, 5, 6]), np.array([4, 5, 6]), kymos[1], "red", line_times[1])
    k5 = KymoTrack(np.array([7]), np.array([7]), kymos[1], "red", line_times[1])

    # Normal use case
    dwell, obs_min, obs_max, removed, dt = KymoTrackGroup._extract_dwelltime_data_from_groups(
        [KymoTrackGroup([k1, k2]), KymoTrackGroup([k3, k4])],
        False,
        observed_minimum=observed_minimum,
    )
    assert removed is False
    np.testing.assert_allclose(dwell, [0.002, 0.003, 0.02, 0.02])
    np.testing.assert_allclose(obs_min, ref_minima)
    np.testing.assert_allclose(obs_max, [0.01, 0.01, 0.1, 0.1])
    np.testing.assert_allclose(dt, [line_times[x] for x in (0, 0, 1, 1)])

    # Drop one "empty" dwell
    dwell, obs_min, obs_max, removed, dt = KymoTrackGroup._extract_dwelltime_data_from_groups(
        [KymoTrackGroup([k1, k2]), KymoTrackGroup([k3, k4, k5])],
        False,
        observed_minimum=observed_minimum,
    )
    np.testing.assert_allclose(dwell, [0.002, 0.003, 0.02, 0.02])
    np.testing.assert_allclose(obs_min, ref_minima)
    np.testing.assert_allclose(obs_max, [0.01, 0.01, 0.1, 0.1])
    assert removed is True
    np.testing.assert_allclose(dt, [line_times[x] for x in (0, 0, 1, 1)])

    # Test with empty group
    dwell, obs_min, obs_max, removed, dt = KymoTrackGroup._extract_dwelltime_data_from_groups(
        [KymoTrackGroup([k1, k2]), KymoTrackGroup([])], False, observed_minimum=observed_minimum
    )
    np.testing.assert_allclose(dwell, [0.002, 0.003])
    np.testing.assert_allclose(obs_min, ref_minima[:2])
    np.testing.assert_allclose(obs_max, [0.01, 0.01])
    assert removed is False
    np.testing.assert_allclose(dt, [line_times[x] for x in (0, 0)])

    # Test with group that is empty after filtering
    dwell, obs_min, obs_max, removed, dt = KymoTrackGroup._extract_dwelltime_data_from_groups(
        [KymoTrackGroup([k1, k2]), KymoTrackGroup([k5])], False, observed_minimum=observed_minimum
    )
    np.testing.assert_allclose(dwell, [0.002, 0.003])
    np.testing.assert_allclose(obs_min, ref_minima[:2])
    np.testing.assert_allclose(obs_max, [0.01, 0.01])
    assert removed is True
    np.testing.assert_allclose(dt, [line_times[x] for x in (0, 0)])

    dwell, obs_min, obs_max, removed, dt = KymoTrackGroup._extract_dwelltime_data_from_groups(
        [KymoTrackGroup([k5]), KymoTrackGroup([k5])], False, observed_minimum=observed_minimum
    )
    np.testing.assert_allclose(dwell, [])
    np.testing.assert_allclose(obs_min, [])
    np.testing.assert_allclose(obs_max, [])
    assert removed is True
    np.testing.assert_allclose(dt, [])


def test_missing_minimum_time(blank_kymo):
    k_missing = KymoTrack(np.array([1, 2, 3]), np.array([1, 2, 3]), blank_kymo, "red", None)
    k_ok = KymoTrack(
        np.array([1, 2, 3]), np.array([1, 2, 3]), blank_kymo, "red", blank_kymo.line_time_seconds
    )
    with pytest.raises(
        RuntimeError, match="Minimum observation time unavailable in KymoTrackGroup"
    ):
        KymoTrackGroup([k_missing, k_ok]).fit_binding_times(
            1, observed_minimum=False, discrete_model=True
        )


@pytest.mark.filterwarnings(
    "ignore:Prior to version 1.2.0 the method `fit_binding_times` had an issue that "
    "assumed that the shortest track duration is indicative of the minimum observable dwell time"
)
@pytest.mark.parametrize(
    "num_samples,ref_amp,ref_lifetime,ref_amp_legacy,ref_lifetime_legacy",
    [
        (
            5000,
            (0.45196325, 0.54803675),
            (1.90610454, 5.84528685),
            (0.45196325, 0.54803675),
            (1.90610454, 5.84528685),
        ),
        (
            20,
            (0.65421428, 0.34578572),
            (2.08205178, 10.81570587),
            (0.59851704, 0.40148296),
            (1.55933902, 9.55647415),
        ),
    ],
)
def test_multi_kymo_dwelltime_analysis(
    simulate_dwelltimes, num_samples, ref_amp, ref_lifetime, ref_amp_legacy, ref_lifetime_legacy
):
    """This test tests only the happy path since others test more specific edge cases"""
    np.random.seed(1337)

    shared_args = {
        "image": np.empty((1, 5000, 3)),
        "color_format": "rgb",
        "start": np.int64(20e9),
        "pixel_size_um": 2,
    }

    kymo1 = _kymo_from_array(**shared_args, line_time_seconds=0.01)
    kymo2 = _kymo_from_array(**shared_args, line_time_seconds=0.02)
    args = [np.array([1.0, 1.0]), kymo1, "red", kymo1.line_time_seconds]

    group1 = KymoTrackGroup(
        [
            KymoTrack(np.array([1, int(round(t / kymo1.line_time_seconds)) + 1]), *args)
            for t in simulate_dwelltimes(2, num_samples, min_time=kymo1.line_time_seconds)
        ],
    )
    args = [np.array([1.0, 1.0]), kymo2, "red", kymo2.line_time_seconds]
    group2 = KymoTrackGroup(
        [
            KymoTrack(np.array([1, int(round(t / kymo2.line_time_seconds)) + 1]), *args)
            for t in simulate_dwelltimes(6, num_samples, min_time=kymo2.line_time_seconds)
        ],
    )

    for observed_minimum in (None, True):
        model = (group1 + group2).fit_binding_times(
            2, tol=1e-16, observed_minimum=observed_minimum, discrete_model=False
        )
        np.testing.assert_allclose(model.lifetimes, ref_lifetime_legacy)
        np.testing.assert_allclose(model.amplitudes, ref_amp_legacy)

    model = (group1 + group2).fit_binding_times(
        2, tol=1e-16, observed_minimum=False, discrete_model=False
    )
    np.testing.assert_allclose(model.lifetimes, ref_lifetime)
    np.testing.assert_allclose(model.amplitudes, ref_amp)


@pytest.mark.parametrize("method,max_lags", [("ols", 2), ("ols", None), ("gls", 2), ("gls", None)])
def test_kymotrack_group_diffusion(blank_kymo, method, max_lags):
    """Tests whether we can call this function at the diffusion level"""
    kymotracks = KymoTrackGroup(
        [
            KymoTrack(time_idx, coordinate, blank_kymo, "red", 0)
            for (time_idx, coordinate) in (
                (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]) / 2),
                (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]) / 3),
                (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]) / 5),
            )
        ]
    )

    for est, kymotrack in zip(kymotracks.estimate_diffusion(method="ols"), kymotracks):
        diff_result = kymotrack.estimate_diffusion(method="ols")
        assert est.method == diff_result.method
        assert est.unit == diff_result.unit
        np.testing.assert_allclose(float(est), float(diff_result.value))
        for attr in ("value", "std_err", "num_lags", "num_points"):
            np.testing.assert_allclose(getattr(est, attr), getattr(diff_result, attr))


def test_disallowed_diffusion_est(blank_kymo):
    contiguous_diffusion_error = (
        "Estimating diffusion constants from data which has been integrated over disjoint "
        "sections of time is not supported. To estimate diffusion constants, do not "
        "downsample the kymograph temporally prior to tracking."
    )

    blank_kymo._contiguous = False
    k = KymoTrack([0, 1, 2, 3, 4, 5], [0.0, 1.0, 1.5, 2.0, 2.5, 3.0], blank_kymo, "red", 0)

    with pytest.raises(NotImplementedError, match=contiguous_diffusion_error):
        k.estimate_diffusion(method="ols")

    group = KymoTrackGroup([k])
    with pytest.raises(NotImplementedError, match=contiguous_diffusion_error):
        group.estimate_diffusion(method="ols")


@pytest.mark.parametrize(
    "localization_var, var_of_localization_var, diffusion_ref, std_err_ref, count_ref,"
    "localization_variance_ref",
    [
        (None, None, 1.5, 1.3928388277184118, 5, -1.0),
        (0.1, 0.01, 0.4, 0.33466401061363027, 5, 0.1),
    ],
)
def test_diffusion_cve(
    blank_kymo,
    blank_kymo_track_args,
    localization_var,
    var_of_localization_var,
    diffusion_ref,
    std_err_ref,
    count_ref,
    localization_variance_ref,
):
    """Test the API for the covariance based estimator"""
    k = KymoTrack(np.arange(5), np.arange(5), *blank_kymo_track_args)
    cve_est = k.estimate_diffusion(
        "cve",
        localization_variance=localization_var,
        variance_of_localization_variance=var_of_localization_var,
    )
    np.testing.assert_allclose(cve_est.value, diffusion_ref)
    np.testing.assert_allclose(cve_est.std_err, std_err_ref)
    np.testing.assert_allclose(cve_est.num_points, count_ref)
    np.testing.assert_allclose(cve_est.localization_variance, localization_variance_ref)

    assert cve_est.num_lags is None
    assert cve_est.method == "cve"
    assert cve_est.unit == "um^2 / s"
    assert cve_est._unit_label == "μm²/s"


def test_diffusion_invalid_loc_variance(blank_kymo, blank_kymo_track_args):
    """In some cases, specifying a localization variance is invalid"""
    track = KymoTrack([1, 2, 3], [1, 2, 3], *blank_kymo_track_args)
    with pytest.raises(
        NotImplementedError,
        match="Passing in a localization error is only supported for method=`cve`",
    ):
        track.estimate_diffusion(
            "ols", localization_variance=1, variance_of_localization_variance=1
        )

    with pytest.raises(
        ValueError,
        match="When the localization variance is provided, the variance of this estimate should "
        "also be provided",
    ):
        track.estimate_diffusion("cve", localization_variance=1)


def test_kymotrack_group_diffusion_filter():
    """Tests whether we can call this function at the diffusion level"""
    image = np.random.randint(0, 20, size=(10, 10, 3))
    kwargs = dict(line_time_seconds=10e-3, start=np.int64(20e9), pixel_size_um=0.05, name="test")
    kymo = _kymo_from_array(image, "rgb", **kwargs)
    kymo._motion_blur_constant = 0

    base_coordinates = (
        np.arange(1, 10),
        np.array([-1.0, 1.0, -1.0, -3.0, -5.0, -1.0, 1.0, -1.0, -3.0, -5.0]),
    )

    def make_coordinates(length, divisor):
        t, p = [c[:length] for c in base_coordinates]
        return t, p / divisor

    tracks = KymoTrackGroup(
        [
            KymoTrack(time_idx, coordinate, kymo, "red", kymo.line_time_seconds)
            for (time_idx, coordinate) in (
                make_coordinates(5, 2),
                make_coordinates(3, 3),
                make_coordinates(2, 5),
                make_coordinates(9, 1),
                make_coordinates(8, 8),
                make_coordinates(2, 4),
            )
        ]
    )

    good_tracks = KymoTrackGroup([track for j, track in enumerate(tracks) if j in (0, 3, 4)])

    warning_string = lambda n_discarded: (
        f"{n_discarded} tracks were shorter than the specified min_length "
        "and discarded from the analysis."
    )

    # test algorithms with default min_length
    with pytest.warns(RuntimeWarning, match=warning_string(3)):
        d = tracks.estimate_diffusion("ols")
        assert len(d) == 3

    with pytest.warns(RuntimeWarning, match=warning_string(3)):
        d = tracks.estimate_diffusion("gls")
        assert len(d) == 3

    with pytest.warns(RuntimeWarning, match=warning_string(2)):
        d = tracks.estimate_diffusion("cve")
        assert len(d) == 4

    # test proper tracks are discarded
    ref_d = [diff.value for diff in good_tracks.estimate_diffusion("ols")]
    with pytest.warns(RuntimeWarning, match=warning_string(3)):
        d = [diff.value for diff in tracks.estimate_diffusion("ols")]
        np.testing.assert_allclose(d, ref_d)

    # test proper forwarding of arguments
    # don't warn if min_length is supplied
    with pytest.warns(RuntimeWarning, match=warning_string(3)):
        d = tracks.estimate_diffusion("ols", max_lag=5)
        assert len(d) == 3

    d = tracks.estimate_diffusion("ols", max_lag=5, min_length=6)
    assert len(d) == 2

    d = tracks.estimate_diffusion("ols", 5, min_length=6)
    assert len(d) == 2

    d = tracks.estimate_diffusion("ols", min_length=6)
    assert len(d) == 2

    # test empty result
    d = tracks.estimate_diffusion("ols", min_length=200)
    assert d == []

    # test throw on invalid min_length
    with pytest.raises(
        RuntimeError,
        match="You need at least 5 time points to estimate the number of points to include in the fit.",
    ):
        d = tracks.estimate_diffusion("ols", min_length=2)


@pytest.mark.parametrize("data", [[], [1, 2, 3, 4]])
def test_ols_empty_kymotrack(blank_kymo, data):
    track = KymoTrack(data, data, blank_kymo, "red", 0)

    with pytest.raises(
        RuntimeError,
        match="You need at least 5 time points to estimate the number of points to include in the "
        "fit.",
    ):
        track.estimate_diffusion("ols")


@pytest.mark.parametrize("kbp_calibration, line_width", [(None, 7), (4, 7), (None, 8)])
def test_ensemble_msd_calibration_from_kymo(blank_kymo, kbp_calibration, line_width):
    """Checks whether all the properties are correctly forwarded from the Kymo"""
    samples_per_pixel, dt, calibration_um = 10, np.int64(1e7), 2
    kymo = generate_kymo(
        "",
        np.ones((line_width, 10)),
        pixel_size_nm=calibration_um * 1000,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=0,
    )

    if kbp_calibration:
        kymo = kymo.calibrate_to_kbp(kbp_calibration)

    space_calibration = kbp_calibration / line_width / calibration_um if kbp_calibration else 1
    frame = np.arange(1, 6)
    tracks = KymoTrackGroup(
        [
            KymoTrack(t, coordinate, kymo, "red", 0)
            for (t, coordinate) in ((frame, frame + 0.1), (frame, frame - 0.1), (frame, frame))
        ]
    )

    line_seconds = line_width * samples_per_pixel * dt / 1e9
    result = tracks.ensemble_msd()
    np.testing.assert_allclose(result.seconds, line_seconds * frame[:-1])
    np.testing.assert_allclose(
        result.msd, (frame[:-1] * calibration_um) ** 2 * space_calibration**2
    )
    np.testing.assert_allclose(result.sem, np.zeros(len(frame) - 1), atol=1e-14)  # zero
    np.testing.assert_allclose(result._time_step, line_seconds)
    np.testing.assert_allclose(result.effective_sample_size, np.ones(len(frame) - 1) * 3)

    assert result.unit == "kbp^2" if kbp_calibration else "um^2"
    assert result._unit_label == "kbp²" if kbp_calibration else "um²"


def test_ensemble_api(blank_kymo):
    """Test whether API arguments are forwarded"""
    short_tracks = [
        KymoTrack(np.arange(1, 6), np.arange(1, 6), blank_kymo, "red", 0) for _ in range(3)
    ]
    long_tracks = [
        KymoTrack(np.arange(1, 7), np.arange(1, 7), blank_kymo, "red", 0) for _ in range(2)
    ]
    tracks = KymoTrackGroup(short_tracks + long_tracks)

    assert len(tracks.ensemble_msd(3).lags) == 3
    assert len(tracks.ensemble_msd(100, 3).lags) == 4
    assert len(tracks.ensemble_msd(100, 2).lags) == 5

    # Because of the gaps in this track, we will be missing lags 1 and 3
    gap_tracks = [
        KymoTrack(np.array([1, 3, 5]), np.array([1, 3, 5]), blank_kymo, "red", 0) for _ in range(3)
    ]
    tracks = KymoTrackGroup(short_tracks[0:2] + gap_tracks)
    np.testing.assert_allclose(tracks.ensemble_msd(100, 3).lags, [2, 4])
    np.testing.assert_allclose(tracks.ensemble_msd(100, 3).msd, [4.0, 16.0])
    np.testing.assert_allclose(tracks.ensemble_msd(100, 2).lags, [1, 2, 3, 4])
    np.testing.assert_allclose(tracks.ensemble_msd(100, 2).msd, [1.0, 4.0, 9.0, 16.0])


def test_ensemble_cve(blank_kymo):
    """Tests whether we can call this function at the diffusion level"""
    kymotracks = KymoTrackGroup(
        [
            KymoTrack(time_idx, coordinate, blank_kymo, "red", 0)
            for (time_idx, coordinate) in (
                (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]) / 2),
                (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]) / 3),
                (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]) / 5),
            )
        ]
    )

    ensemble_diffusion = kymotracks.ensemble_diffusion("cve")
    np.testing.assert_allclose(ensemble_diffusion.value, 0.445679012345679)
    np.testing.assert_allclose(ensemble_diffusion.std_err, 0.20555092123942093)
    np.testing.assert_allclose(ensemble_diffusion.localization_variance, -0.1782716049382716)
    np.testing.assert_allclose(
        ensemble_diffusion.variance_of_localization_variance, 0.006760188995579941
    )
    np.testing.assert_allclose(ensemble_diffusion.num_points, 15)
    assert ensemble_diffusion.method == "ensemble cve"
    assert ensemble_diffusion.unit == "um^2 / s"
    assert ensemble_diffusion._unit_label == "um^2 / s"

    # Consistency check
    single_group_msd = KymoTrackGroup([kymotracks[0]]).ensemble_diffusion("cve")
    single_track_msd = kymotracks[0].estimate_diffusion("cve")
    np.testing.assert_allclose(single_group_msd.value, single_track_msd.value)
    np.testing.assert_allclose(single_group_msd.std_err, single_track_msd.std_err)


@pytest.mark.parametrize(
    "max_lag, diffusion_ref, std_err_ref, localization_var_ref",
    [
        (None, 0.44567901234567886, 0.27564925652921307, -0.17827160493827154),
        (4, 0.3030617283950619, 0.2895503367634419, 0.08913580246913569),
    ],
)
def test_ensemble_ols(blank_kymo, max_lag, diffusion_ref, std_err_ref, localization_var_ref):
    """Tests the ensemble diffusion estimate"""
    kymotracks = KymoTrackGroup(
        [
            KymoTrack(time_idx, coordinate, blank_kymo, "red", 0)
            for (time_idx, coordinate) in (
                (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]) / 2),
                (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]) / 3),
                (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]) / 5),
            )
        ]
    )

    ensemble_diffusion = kymotracks.ensemble_diffusion("ols", max_lag=max_lag)
    np.testing.assert_allclose(ensemble_diffusion.value, diffusion_ref)
    np.testing.assert_allclose(ensemble_diffusion.std_err, std_err_ref)
    np.testing.assert_allclose(ensemble_diffusion.localization_variance, localization_var_ref)
    assert ensemble_diffusion.variance_of_localization_variance is None
    np.testing.assert_allclose(ensemble_diffusion.num_points, 15)
    assert ensemble_diffusion.method == "ensemble ols"
    assert ensemble_diffusion.unit == "um^2 / s"
    assert ensemble_diffusion._unit_label == "μm²/s"


@pytest.mark.parametrize("max_lag", [None, 4])
def test_ensemble_ols_multiple_sources(blank_kymo, max_lag):
    """Tests the happy path for OLS ensemble diffusion estimate with multiple source kymos."""
    groups = [
        KymoTrackGroup(
            [
                KymoTrack(time_idx, coordinate, kymo, "red", 0)
                for (time_idx, coordinate) in (
                    (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]) / 2),
                    (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]) / 3),
                    (np.arange(1, 6), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]) / 5),
                )
            ]
        )
        for kymo in (blank_kymo, copy(blank_kymo))
    ]
    assert id(groups[0]._kymos[0]) != id(groups[1]._kymos[0])

    tracks = groups[0][:2] + groups[1][2]
    assert len(tracks._kymos) == 2

    ref_ensemble_diffusion = groups[0].ensemble_diffusion("ols", max_lag=max_lag)
    ensemble_diffusion = tracks.ensemble_diffusion("ols", max_lag=max_lag)

    np.testing.assert_allclose(ensemble_diffusion.value, ref_ensemble_diffusion.value)
    np.testing.assert_allclose(ensemble_diffusion.std_err, ref_ensemble_diffusion.std_err)
    np.testing.assert_allclose(
        ensemble_diffusion.localization_variance, ref_ensemble_diffusion.localization_variance
    )
    assert ensemble_diffusion.variance_of_localization_variance is None
    np.testing.assert_allclose(ensemble_diffusion.num_points, 15)
    assert ensemble_diffusion.method == "ensemble ols"
    assert ensemble_diffusion.unit == "um^2 / s"
    assert ensemble_diffusion._unit_label == "μm²/s"


def test_invalid_ensemble_diffusion(blank_kymo):
    """Tests whether we can call this function at the diffusion level"""
    kymotracks = KymoTrackGroup([KymoTrack([], [], blank_kymo, "red", 0)])
    with pytest.raises(ValueError, match=re.escape("Invalid method (egg) selected")):
        kymotracks.ensemble_diffusion("egg")


def test_ensemble_diffusion_different_attributes():
    line_times = (1, 0.5)
    pixel_sizes = (0.1, 0.05)
    multi_kwargs = [
        {"line_time_seconds": t, "pixel_size_um": s} for t in line_times for s in pixel_sizes
    ]
    kymos = []
    for kwargs in multi_kwargs:
        kymo = _kymo_from_array(np.random.poisson(5, (25, 25, 3)), "rgb", **kwargs)
        kymo._motion_blur_constant = 0
        kymos.append(kymo)

    tracks = [
        KymoTrackGroup(
            [KymoTrack(np.arange(5), np.random.uniform(3, 5, 5), k, "green", 0) for _ in range(5)]
        )
        for k in kymos
    ]

    error_messages = {
        "line times": re.escape(
            "All source kymographs must have the same line times, got [0.5, 1] seconds."
        ),
        "pixel sizes": re.escape(
            "All source kymographs must have the same pixel sizes, got [0.05, 0.1] um."
        ),
        "both": re.escape(
            "All source kymographs must have the same line times, got [0.5, 1] seconds. "
            "All source kymographs must have the same pixel sizes, got [0.05, 0.1] um."
        ),
    }

    combo_tracks = tracks[0] + tracks[1]
    with pytest.raises(ValueError, match=error_messages["pixel sizes"]):
        combo_tracks.ensemble_msd(max_lag=3)
    with pytest.raises(ValueError, match=error_messages["pixel sizes"]):
        combo_tracks.ensemble_diffusion(method="ols")

    combo_tracks = tracks[0] + tracks[2]
    with pytest.raises(ValueError, match=error_messages["line times"]):
        combo_tracks.ensemble_msd(max_lag=3)
    with pytest.raises(ValueError, match=error_messages["line times"]):
        combo_tracks.ensemble_diffusion(method="ols")

    combo_tracks = tracks[0] + tracks[3]
    with pytest.raises(ValueError, match=error_messages["both"]):
        combo_tracks.ensemble_msd(max_lag=3)
    with pytest.raises(ValueError, match=error_messages["both"]):
        combo_tracks.ensemble_diffusion(method="ols")

    with pytest.warns(
        RuntimeWarning,
        match=(
            "Localization variances cannot be reliably calculated for an ensemble of tracks from "
            "kymographs with different line times or pixel sizes."
        ),
    ):
        combo_tracks.ensemble_diffusion(method="cve")


@pytest.mark.parametrize(
    "window, pixelsize, result",
    [
        # fmt:off
        (1.0, 1.0, 0), (2.0, 1.0, 1), (3.0, 1.0, 1), (3.01, 1.0, 2), (4.0, 1.0, 2), (4.99, 1.0, 2),
        (1.0, 2.0, 0), (2.0, 2.0, 0), (6.0, 2.0, 1), (6.01, 2.0, 2), (7.0, 2.0, 2), (7.99, 2.0, 2),
        # fmt:on
    ],
)
def test_half_kernel(window, pixelsize, result):
    assert _to_half_kernel_size(window, pixelsize) == result


def test_integral_times_kymotrack(blank_kymo):
    with pytest.raises(TypeError, match="Time indices should be of integer type, got float64"):
        KymoTrack([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], blank_kymo, "red", 0)


def test_no_motion_blur(blank_kymo):
    """When the motion blur of a Kymograph is unknown, a warning should be issued and
    return values blanked"""
    unknown_blur_kymo = copy(blank_kymo)
    unknown_blur_kymo._motion_blur_constant = None
    track = KymoTrack([1, 2, 3, 4], [1, 2, 3, 2], unknown_blur_kymo, "red", 0)

    with pytest.warns(
        RuntimeWarning,
        match="Motion blur cannot be taken into account for this type of Kymo",
    ):
        estimate = track.estimate_diffusion("cve")
        assert np.isnan(estimate.std_err)
        assert np.isnan(estimate.localization_variance)

    with pytest.raises(
        ValueError,
        match="Cannot compute diffusion constant reliably for a kymograph that does not"
        "have a clearly defined motion blur constant and the localization variance "
        "is provided. Omit the localization variance to calculate a diffusion "
        "constant.",
    ):
        track.estimate_diffusion("cve", localization_variance=1)


def test_photon_counts_api(blank_kymo):
    with pytest.raises(AttributeError, match="Photon counts are unavailable for this KymoTrack."):
        KymoTrack([1, 2, 3], [1, 2, 3], blank_kymo, "red", 0).photon_counts

    np.testing.assert_equal(
        KymoTrack(
            [1, 2, 3],
            GaussianLocalizationModel(
                [1, 2, 3], [1, 2, 5], [1, 1, 1], [0, 0, 0], [False, True, False]
            ),
            blank_kymo,
            "red",
            0,
        ).photon_counts,
        [1, 2, 5],
    )


@pytest.mark.slow
def test_integration_binding_times_with_minimum_time():
    """End-to-end test for dwell time analysis while filtering with a set duration"""
    dt = 0.1
    kymo = _kymo_from_array(np.zeros((1, 10000)), "r", dt)
    params = ExponentialParameters(
        amplitudes=[0.5, 0.5],
        lifetimes=[0.2, 10.0],
        _observation_limits=(0, np.inf),
        dt=dt,
    )

    np.random.seed(10071985)
    data = make_dataset(params, n_samples=100000)["data"]

    tracks = KymoTrackGroup([KymoTrack([0, int(t / dt)], [0, 0], kymo, "red", 0) for t in data])

    # Minimum we filter by is deliberately not a multiple of dt
    tracks = filter_tracks(tracks, minimum_length=0, minimum_duration=0.15)
    model = tracks.fit_binding_times(
        2, exclude_ambiguous_dwells=False, discrete_model=True, observed_minimum=False
    )
    np.testing.assert_allclose(np.sort(model.lifetimes), [0.2, 10.0], rtol=1e-2)
