import re
import pytest
import matplotlib.pyplot as plt
from lumicks.pylake.kymotracker.kymotrack import *
from lumicks.pylake.kymotracker.kymotracker import _to_half_kernel_size
from lumicks.pylake.kymotracker.detail.localization_models import *
from lumicks.pylake import filter_tracks
from lumicks.pylake.kymo import _kymo_from_array
from ...tests.data.mock_confocal import generate_kymo


def test_kymo_track(blank_kymo):
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([2, 3, 4]), blank_kymo, "red")
    np.testing.assert_allclose(k1[1], [2, 3])
    np.testing.assert_allclose(k1[-1], [3, 4])
    np.testing.assert_allclose(k1[0:2], [[1, 2], [2, 3]])
    np.testing.assert_allclose(k1[0:2][:, 1], [2, 3])

    k2 = KymoTrack(np.array([4, 5, 6]), np.array([5, 6, 7]), blank_kymo, "red")
    np.testing.assert_allclose((k1 + k2)[:], [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    np.testing.assert_allclose(k1.extrapolate(True, 3, 2.0), [5, 6])

    # Need at least 2 points for linear extrapolation
    with pytest.raises(AssertionError):
        KymoTrack([1], [1], blank_kymo, "red").extrapolate(True, 5, 2.0)

    with pytest.raises(AssertionError):
        KymoTrack([1, 2, 3], [1, 2, 3], blank_kymo, "red").extrapolate(True, 1, 2.0)

    k1 = KymoTrack([1, 2, 3], [1, 2, 3], blank_kymo, "red")
    k2 = k1.with_offset(2, 2)
    assert id(k2) != id(k1)

    np.testing.assert_allclose(k2.coordinate_idx, [3, 4, 5])
    assert k1._kymo == blank_kymo
    assert k2._kymo == blank_kymo
    assert k1._channel == "red"
    assert k2._channel == "red"


def test_kymotrack_selection(blank_kymo):
    t = np.array([4, 5, 6])
    y = np.array([7, 7, 7])
    assert not KymoTrack(t, y, blank_kymo, "red").in_rect(((4, 6), (6, 7)))
    assert KymoTrack(t, y, blank_kymo, "red").in_rect(((4, 6), (6, 8)))
    assert not KymoTrack(t, y, blank_kymo, "red").in_rect(((3, 6), (4, 8)))
    assert KymoTrack(t, y, blank_kymo, "red").in_rect(((3, 6), (5, 8)))

    assert KymoTrack([2], [6], blank_kymo, "red").in_rect(((2, 5), (3, 8)))
    assert KymoTrack([2], [5], blank_kymo, "red").in_rect(((2, 5), (3, 8)))
    assert not KymoTrack([4], [6], blank_kymo, "red").in_rect(((2, 5), (3, 8)))
    assert not KymoTrack([1], [6], blank_kymo, "red").in_rect(((2, 5), (3, 8)))
    assert not KymoTrack([2], [4], blank_kymo, "red").in_rect(((2, 5), (3, 8)))
    assert not KymoTrack([2], [8], blank_kymo, "red").in_rect(((2, 5), (3, 8)))


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
    assert not KymoTrack(time, pos, kymo, "red").in_rect(((4, 6 * 5), (6, 7 * 5)))
    assert KymoTrack(time, pos, kymo, "red").in_rect(((4, 6 * 5), (6, 8 * 5)))


def test_kymotracks_removal(blank_kymo):
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([1, 1, 1]), blank_kymo, "red")
    k2 = KymoTrack(np.array([2, 3, 4]), np.array([2, 2, 2]), blank_kymo, "red")
    k3 = KymoTrack(np.array([3, 4, 5]), np.array([3, 3, 3]), blank_kymo, "red")

    def verify(rect, resulting_tracks):
        k = KymoTrackGroup([k1, k2, k3])
        k.remove_tracks_in_rect(rect)
        assert len(k._src) == len(resulting_tracks)
        assert all([l1 == l2 for l1, l2 in zip(k._src, resulting_tracks)])

    verify([[5, 3], [6, 4]], [k1, k2])
    verify([[6, 3], [5, 4]], [k1, k2])
    verify([[6, 5], [5, 3]], [k1, k2])
    verify([[0, 0], [5, 5]], [])
    verify([[15, 3], [16, 4]], [k1, k2, k3])

    with pytest.warns(DeprecationWarning):
        k = KymoTrackGroup([k1, k2, k3])
        k.remove_lines_in_rect([[5, 3], [6, 4]])


def test_kymotrackgroup(blank_kymo):
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([2, 3, 4]), blank_kymo, "red")
    k2 = KymoTrack(np.array([2, 3, 4]), np.array([3, 4, 5]), blank_kymo, "red")
    k3 = KymoTrack(np.array([3, 4, 5]), np.array([4, 5, 6]), blank_kymo, "red")
    k4 = KymoTrack(np.array([4, 5, 6]), np.array([5, 6, 7]), blank_kymo, "red")

    tracks = KymoTrackGroup([k1, k2, k3, k4])
    assert [k for k in tracks] == [k1, k2, k3, k4]
    assert len(tracks) == 4
    assert tracks[0] == k1
    assert tracks[1] == k2
    assert tracks[0:2][0] == k1
    assert tracks[0:2][1] == k2

    with pytest.raises(IndexError):
        tracks[0:2][2]

    with pytest.raises(NotImplementedError):
        tracks[1] = 4

    tracks = KymoTrackGroup([k1, k2])
    tracks.extend(KymoTrackGroup([k3, k4]))
    assert [k for k in tracks] == [k1, k2, k3, k4]

    tracks = KymoTrackGroup([k1, k2, k3])
    tracks.extend(k4)
    assert [k for k in tracks] == [k1, k2, k3, k4]

    with pytest.raises(TypeError):
        tracks.extend(5)


def test_kymotrackgroup(blank_kymo):
    def validate_same(kymoline_group, ref_list, source_items, ref_kymo):
        assert [k for k in kymoline_group] == ref_list
        assert id(kymoline_group) not in (id(s) for s in source_items)
        if ref_kymo:
            assert id(kymoline_group._kymo) == id(ref_kymo)

    k1 = KymoTrack(np.array([1, 2, 3]), np.array([2, 3, 4]), blank_kymo, "red")
    k2 = KymoTrack(np.array([2, 3, 4]), np.array([3, 4, 5]), blank_kymo, "red")
    k3 = KymoTrack(np.array([3, 4, 5]), np.array([4, 5, 6]), blank_kymo, "red")
    k4 = KymoTrack(np.array([4, 5, 6]), np.array([5, 6, 7]), blank_kymo, "red")

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


def test_kymotrack_concat(blank_kymo):
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([1, 1, 1]), blank_kymo, "red")
    k2 = KymoTrack(np.array([6, 7, 8]), np.array([2, 2, 2]), blank_kymo, "red")
    k3 = KymoTrack(np.array([3, 4, 5]), np.array([3, 3, 3]), blank_kymo, "red")
    k4 = KymoTrack(np.array([8, 9, 10]), np.array([3, 3, 3]), blank_kymo, "red")
    group = KymoTrackGroup([k1, k2, k3])

    # Test whether overlapping time raises.
    with pytest.raises(RuntimeError):
        group._concatenate_tracks(k1, k3)

    # Kymotracks have to be added sequentially. Check whether the wrong order raises.
    with pytest.raises(RuntimeError):
        group._concatenate_tracks(k2, k1)

    # Check whether a track that's not in the group raises (should only be able to merge
    # within group)
    with pytest.raises(RuntimeError):
        group._concatenate_tracks(k1, k4)

    # Check whether a track that's not in the group raises (should only be able to merge
    # within group)
    with pytest.raises(RuntimeError):
        group._concatenate_tracks(k4, k1)

    # Check whether an invalid type raises correctly
    with pytest.raises(RuntimeError):
        group._concatenate_tracks(k1, 5)

    # Finally do a correct merge
    group._concatenate_tracks(k1, k2)
    np.testing.assert_allclose(group[0].seconds, [1, 2, 3, 6, 7, 8])
    np.testing.assert_allclose(group[1].seconds, [3, 4, 5])


def test_kymotrack_merge():
    image = np.random.randint(0, 20, size=(10, 10, 3))
    kwargs = dict(line_time_seconds=10e-3, start=np.int64(20e9), pixel_size_um=0.05, name="test")
    kymo = _kymo_from_array(image, "rgb", **kwargs)

    time_idx = ([1, 2, 3, 4, 5], [6, 7, 8], [6, 7, 8], [1, 2, 3])
    pos_idx = ([1, 1, 1, 3, 3], [4, 4, 4], [9, 9, 9], [10, 10, 10])

    make_tracks = lambda: KymoTrackGroup(
        [KymoTrack(t, p, kymo, "green") for t, p in zip(time_idx, pos_idx)]
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
    tracks2 = KymoTrackGroup([KymoTrack([1, 2, 3], [4, 5, 6], kymo, "green")])
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

    # can't connect nodes with same time index
    tracks = make_tracks()
    with pytest.raises(AssertionError, match="Cannot connect two points with the same time index."):
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
    k = KymoTrack(time_scale * time_idx, position_scale * position_idx, kymo, "red")

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
    k = KymoTrack(time_idx, coordinate / pixel_size, kymo, "red")

    np.testing.assert_allclose(k.estimate_diffusion("ols", max_lag=max_lag).value, diffusion_const)


@pytest.mark.parametrize("calibration_coeff", [0.5, 2.0])
def test_diffusion_units(blank_kymo, calibration_coeff):
    kymotrack, kymotrack_kbp = [
        KymoTrack(
            np.arange(1, 6),
            np.array([-1.0, 1.0, -1.0, -3.0, -5.0]),
            kymo,
            "red",
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
    assert diffusion_estimate._unit_label == "$\\mu$m$^2$/s"


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
    k = KymoTrack(time_idx, coordinate / pixel_size, kymo, "red")

    np.testing.assert_allclose(k.estimate_diffusion("gls", max_lag=max_lag).value, diffusion_const)


def test_invalid_method(blank_kymo):
    k = KymoTrack(np.arange(5), np.arange(5), blank_kymo, "red")
    with pytest.raises(ValueError, match="Invalid method selected"):
        k.estimate_diffusion(max_lag=5, method="BAD")


def test_lag_default(blank_kymo):
    """Checks whether the max_lag argument is correctly set by default"""
    k = KymoTrack(np.arange(5), np.arange(5), blank_kymo, "red")
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
    k = KymoTrack(np.arange(1, 6), np.arange(1, 6), kymo, "red")
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

    k1 = KymoTrack(np.array([1, 2, 3]), np.array([2.5, 3.5, 4.5]), kymo, "red")
    k2 = KymoTrack(np.array([2, 3, 4]), np.array([3.5, 4.5, 5.5]), kymo, "red")
    k3 = KymoTrack(np.array([3, 4, 5]), np.array([4.5, 5.5, 6.5]), kymo, "red")
    k4 = KymoTrack(np.array([4, 5, 6]), np.array([5.5, 6.5, 7.5]), kymo, "red")

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


def test_kymotrackgroup_source_kymo():
    # test that all tracks are from the same source Kymo and tracked
    # on the same color channel

    image = np.random.randint(0, 20, size=(10, 10, 3))
    kwargs = dict(line_time_seconds=10e-3, start=np.int64(20e9), pixel_size_um=0.05, name="test")
    kymos = [_kymo_from_array(image, "rgb", **kwargs) for _ in range(2)]

    time_idx = ([1, 2, 3], [4, 6, 7], [1, 2, 3], [4, 6, 7])

    pos_idx = ([4, 5, 6], [1, 7, 7], [1, 2, 3], [1, 2, 3])

    green_tracks_a = [KymoTrack(t, p, kymos[0], "green") for t, p in zip(time_idx, pos_idx)]
    green_tracks_b = [KymoTrack(t, p, kymos[1], "green") for t, p in zip(time_idx, pos_idx)]

    red_tracks_a = [KymoTrack(t, p, kymos[0], "red") for t, p in zip(time_idx, pos_idx)]
    red_tracks_b = [KymoTrack(t, p, kymos[1], "red") for t, p in zip(time_idx, pos_idx)]

    # test proper group construction
    tracks_a = KymoTrackGroup(green_tracks_a[:2])
    assert len(tracks_a) == 2

    tracks_b = KymoTrackGroup(green_tracks_b[:2])
    assert len(tracks_b) == 2

    assert id(tracks_a._kymo) == id(kymos[0])
    assert tracks_a._channel == "green"

    # test empty result
    tracks_empty = KymoTrackGroup([])
    assert len(tracks_empty) == 0

    tracks_empty = filter_tracks(tracks_a, 5)
    assert len(tracks_empty) == 0

    with pytest.raises(
        RuntimeError,
        match=re.escape("No kymo associated with this empty group (no tracks available)"),
    ):
        tracks_empty._kymo

    with pytest.raises(
        RuntimeError,
        match=re.escape("No channel associated with this empty group (no tracks available)"),
    ):
        tracks_empty._channel

    # cannot make group from different source kymos
    with pytest.raises(AssertionError, match="All tracks must have the same source kymograph."):
        KymoTrackGroup([*green_tracks_a, *green_tracks_b])

    # test extend with single track
    tracks_a.extend(green_tracks_a[2])
    assert len(tracks_a) == 3

    # test extend with KymoTrackGroup
    tracks_a.extend(KymoTrackGroup(green_tracks_a[-1:]))
    assert len(tracks_a) == 4

    # cannot extend with different source kymos
    with pytest.raises(AssertionError, match="All tracks must have the same source kymograph."):
        tracks_a.extend(tracks_b)

    # test extend from empty group
    tracks_empty = KymoTrackGroup([])
    tracks_empty.extend(tracks_a)
    assert len(tracks_empty) == 4

    # cannot make group from different color channels
    with pytest.raises(AssertionError, match="All tracks must be from the same color channel."):
        KymoTrackGroup([*green_tracks_a, *red_tracks_a])

    # cannot extend with different color channels
    with pytest.raises(AssertionError, match="All tracks must be from the same color channel."):
        tracks_a.extend(red_tracks_a[0])
    with pytest.raises(AssertionError, match="All tracks must be from the same color channel."):
        tracks_a.extend(KymoTrackGroup(red_tracks_a))


def test_kymotrackgroup_copy(blank_kymo):
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([1, 1, 1]), blank_kymo, "red")
    k2 = KymoTrack(np.array([6, 7, 8]), np.array([2, 2, 2]), blank_kymo, "red")
    group = KymoTrackGroup([k1, k2])
    assert id(group._src) != id(copy(group)._src)


def test_kymotrack_concat_gaussians(blank_kymo):
    k1 = KymoTrack(np.array([1, 2, 3]), np.array([1, 1, 1]), blank_kymo, "red")
    k2 = KymoTrack(
        np.array([6, 7, 8]),
        GaussianLocalizationModel(
            np.full(3, 2), np.full(3, 7), np.full(3, 0.5), np.ones(3), np.full(3, False)
        ),
        blank_kymo,
        "red",
    )
    group = KymoTrackGroup([k1, k2])

    assert isinstance(k1._localization, LocalizationModel)
    assert isinstance(k2._localization, GaussianLocalizationModel)

    # test concatenation clears gaussian parameters
    group._concatenate_tracks(k1, k2)
    assert len(group) == 1
    assert isinstance(group[0]._localization, LocalizationModel)


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

    k1 = KymoTrack(np.array([1, 2, 3]), np.array([2, 3, 4]), kymo, "red")
    k2 = KymoTrack(np.array([2, 3, 4]), np.array([3, 4, 5]), kymo, "red")
    k3 = KymoTrack(np.array([3, 4, 5]), np.array([4, 5, 6]), kymo, "red")
    k4 = KymoTrack(np.array([4, 5, 6]), np.array([5, 6, 7]), kymo, "red")

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


def test_fit_binding_times(blank_kymo):
    k1 = KymoTrack(np.array([0, 1, 2]), np.zeros(3), blank_kymo, "red")
    k2 = KymoTrack(np.array([2, 3, 4, 5, 6]), np.zeros(5), blank_kymo, "red")
    k3 = KymoTrack(np.array([3, 4, 5]), np.zeros(3), blank_kymo, "red")
    k4 = KymoTrack(np.array([8, 9]), np.zeros(2), blank_kymo, "red")

    tracks = KymoTrackGroup([k1, k2, k3, k4])

    dwells = tracks.fit_binding_times(1)
    np.testing.assert_allclose(dwells.lifetimes, [1.002547])

    dwells = tracks.fit_binding_times(1, exclude_ambiguous_dwells=False)
    np.testing.assert_allclose(dwells.lifetimes, [1.25710457])


def test_fit_binding_times_nonzero(blank_kymo):
    k1 = KymoTrack(np.array([2]), np.zeros(3), blank_kymo, "red")
    k2 = KymoTrack(np.array([2, 3, 4, 5, 6]), np.zeros(5), blank_kymo, "red")
    tracks = KymoTrackGroup([k1, k2, k2, k2, k2])

    with pytest.warns(
        RuntimeWarning,
        match=r"Some dwell times are zero. A dwell time of zero indicates that some of the tracks "
        r"were only observed in a single frame. For these samples it is not possible to "
        r"actually determine a dwell time. Therefore these samples are dropped from the "
        r"analysis. If you wish to not see this warning, filter the tracks with "
        r"`lk.filter_tracks` with a minimum length of 2 samples.",
    ):
        dwelltime_model = tracks.fit_binding_times(1)
        np.testing.assert_equal(dwelltime_model.dwelltimes, [4, 4, 4, 4])
        np.testing.assert_equal(dwelltime_model._observation_limits[0], 4)
        np.testing.assert_allclose(dwelltime_model.lifetimes[0], [0.4])


def test_fit_binding_times_empty():
    with pytest.raises(RuntimeError, match="No tracks available for analysis"):
        KymoTrackGroup([]).fit_binding_times(1)


@pytest.mark.parametrize("method,max_lags", [("ols", 2), ("ols", None), ("gls", 2), ("gls", None)])
def test_kymotrack_group_diffusion(blank_kymo, method, max_lags):
    """Tests whether we can call this function at the diffusion level"""
    kymotracks = KymoTrackGroup(
        [
            KymoTrack(time_idx, coordinate, blank_kymo, "red")
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
    k = KymoTrack([0, 1, 2, 3, 4, 5], [0.0, 1.0, 1.5, 2.0, 2.5, 3.0], blank_kymo, "red")

    with pytest.raises(NotImplementedError, match=contiguous_diffusion_error), pytest.warns(
        DeprecationWarning, match="Call to deprecated method estimate_diffusion_ols"
    ):
        k.estimate_diffusion_ols()

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
    localization_var,
    var_of_localization_var,
    diffusion_ref,
    std_err_ref,
    count_ref,
    localization_variance_ref,
):
    """Test the API for the covariance based estimator"""
    k = KymoTrack(np.arange(5), np.arange(5), blank_kymo, "red")
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
    assert cve_est._unit_label == "$\\mu$m$^2$/s"


def test_diffusion_invalid_loc_variance(blank_kymo):
    """In some cases, specifying a localization variance is invalid"""
    track = KymoTrack([1, 2, 3], [1, 2, 3], blank_kymo, "red")
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

    base_coordinates = (
        np.arange(1, 10),
        np.array([-1.0, 1.0, -1.0, -3.0, -5.0, -1.0, 1.0, -1.0, -3.0, -5.0]),
    )

    def make_coordinates(length, divisor):
        t, p = [c[:length] for c in base_coordinates]
        return t, p / divisor

    tracks = KymoTrackGroup(
        [
            KymoTrack(time_idx, coordinate, kymo, "red")
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
            KymoTrack(t, coordinate, kymo, "red")
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
    assert result._unit_label == "kbp$^2$" if kbp_calibration else "um$^2$"


def test_ensemble_api(blank_kymo):
    """Test whether API arguments are forwarded"""
    track = KymoTrack(np.arange(1, 6), np.arange(1, 6), blank_kymo, "red")
    long_track = KymoTrack(np.arange(1, 7), np.arange(1, 7), blank_kymo, "red")
    tracks = KymoTrackGroup([track, track, track, long_track, long_track])

    assert len(tracks.ensemble_msd(3).lags) == 3
    assert len(tracks.ensemble_msd(100, 3).lags) == 4
    assert len(tracks.ensemble_msd(100, 2).lags) == 5

    # Because of the gaps in this track, we will be missing lags 1 and 3
    gap_track = KymoTrack(np.array([1, 3, 5]), np.array([1, 3, 5]), blank_kymo, "red")
    tracks = KymoTrackGroup([track, track, gap_track, gap_track, gap_track])
    np.testing.assert_allclose(tracks.ensemble_msd(100, 3).lags, [2, 4])
    np.testing.assert_allclose(tracks.ensemble_msd(100, 3).msd, [4.0, 16.0])
    np.testing.assert_allclose(tracks.ensemble_msd(100, 2).lags, [1, 2, 3, 4])
    np.testing.assert_allclose(tracks.ensemble_msd(100, 2).msd, [1.0, 4.0, 9.0, 16.0])


def test_ensemble_cve(blank_kymo):
    """Tests whether we can call this function at the diffusion level"""
    kymotracks = KymoTrackGroup(
        [
            KymoTrack(time_idx, coordinate, blank_kymo, "red")
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
    single_group_msd = KymoTrackGroup([kymotracks._src[0]]).ensemble_diffusion("cve")
    single_track_msd = kymotracks._src[0].estimate_diffusion("cve")
    np.testing.assert_allclose(single_group_msd.value, single_track_msd.value)
    np.testing.assert_allclose(single_group_msd.std_err, single_track_msd.std_err)


@pytest.mark.parametrize("max_lag, diffusion_ref, std_err_ref, localization_var_ref", [
    (None, 0.44567901234567886, 0.27564925652921307, -0.17827160493827154),
    (4, 0.3030617283950619, 0.2895503367634419, 0.08913580246913569),
])
def test_ensemble_ols(blank_kymo, max_lag, diffusion_ref, std_err_ref, localization_var_ref):
    """Tests the ensemble diffusion estimate"""
    kymotracks = KymoTrackGroup(
        [
            KymoTrack(time_idx, coordinate, blank_kymo, "red")
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
    assert ensemble_diffusion._unit_label == "$\\mu$m$^2$/s"


def test_invalid_ensemble_diffusion(blank_kymo):
    """Tests whether we can call this function at the diffusion level"""
    kymotracks = KymoTrackGroup([KymoTrack([], [], blank_kymo, "red")])
    with pytest.raises(ValueError, match=re.escape("Invalid method (egg) selected")):
        kymotracks.ensemble_diffusion("egg")


@pytest.mark.parametrize(
    "window, pixelsize, result",
    [
        # fmt:off
        (1.0, 1.0, 0), (2.0, 1.0, 1), (3.0, 1.0, 1), (3.01, 1.0, 2), (4.0, 1.0, 2), (4.99, 1.0, 2),
        (1.0, 2.0, 0), (2.0, 2.0, 0), (6.0, 2.0, 1), (6.01, 2.0, 2), (7.0, 2.0, 2), (7.99, 2.0, 2),
        # fmt:on
    ]
)
def test_half_kernel(window, pixelsize, result):
    assert _to_half_kernel_size(window, pixelsize) == result
