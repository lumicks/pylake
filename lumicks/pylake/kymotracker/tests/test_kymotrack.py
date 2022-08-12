import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup
from lumicks.pylake.kymotracker.kymotrack import *
from lumicks.pylake.kymotracker.detail.localization_models import *
from lumicks.pylake.tests.data.mock_confocal import generate_kymo


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

    np.testing.assert_allclose(
        k.estimate_diffusion("ols", max_lag=max_lag).value, diffusion_const
    )


@pytest.mark.parametrize("calibration_coeff", [0.5, 2.0])
def test_diffusion_units(blank_kymo, calibration_coeff):
    kymotrack, kymotrack_kbp = [
        KymoTrack(
            np.arange(1, 6),
            np.array([-1.0, 1.0, -1.0, -3.0, -5.0]),
            kymo,
            "red",
        ) for kymo in (blank_kymo, blank_kymo.calibrate_to_kbp(calibration_coeff))]

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
@cleanup
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
    x, densities = tracks._histogram_binding_profile(3, 0.2, 4)

    np.testing.assert_allclose(x, np.linspace(0, 10, 4))
    np.testing.assert_allclose(
        densities[0], [1.28243310e-022, 3.31590463e-001, 1.37463811e-073, 1.31346543e-266]
    )
    np.testing.assert_allclose(
        densities[1], [1.03517782e-87, 2.89177312e-03, 1.92784875e-03, 6.90118545e-88]
    )
    np.testing.assert_allclose(
        densities[2], [1.97019814e-266, 2.06195717e-073, 4.97385694e-001, 2.76535477e-049]
    )

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
