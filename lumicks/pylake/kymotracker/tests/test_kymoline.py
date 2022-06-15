import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup
from lumicks.pylake.kymotracker.kymoline import *
from lumicks.pylake.kymotracker.detail.localization_models import *
from lumicks.pylake.tests.data.mock_confocal import generate_kymo


def test_kymo_line(blank_kymo):
    k1 = KymoLine(np.array([1, 2, 3]), np.array([2, 3, 4]), blank_kymo, "red")
    np.testing.assert_allclose(k1[1], [2, 3])
    np.testing.assert_allclose(k1[-1], [3, 4])
    np.testing.assert_allclose(k1[0:2], [[1, 2], [2, 3]])
    np.testing.assert_allclose(k1[0:2][:, 1], [2, 3])

    k2 = KymoLine(np.array([4, 5, 6]), np.array([5, 6, 7]), blank_kymo, "red")
    np.testing.assert_allclose((k1 + k2)[:], [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    np.testing.assert_allclose(k1.extrapolate(True, 3, 2.0), [5, 6])

    # Need at least 2 points for linear extrapolation
    with pytest.raises(AssertionError):
        KymoLine([1], [1], blank_kymo, "red").extrapolate(True, 5, 2.0)

    with pytest.raises(AssertionError):
        KymoLine([1, 2, 3], [1, 2, 3], blank_kymo, "red").extrapolate(True, 1, 2.0)

    k1 = KymoLine([1, 2, 3], [1, 2, 3], blank_kymo, "red")
    k2 = k1.with_offset(2, 2)
    assert id(k2) != id(k1)

    np.testing.assert_allclose(k2.coordinate_idx, [3, 4, 5])
    assert k1._kymo == blank_kymo
    assert k2._kymo == blank_kymo
    assert k1._channel == "red"
    assert k2._channel == "red"


def test_kymoline_selection(blank_kymo):
    t = np.array([4, 5, 6])
    y = np.array([7, 7, 7])
    assert not KymoLine(t, y, blank_kymo, "red").in_rect(((4, 6), (6, 7)))
    assert KymoLine(t, y, blank_kymo, "red").in_rect(((4, 6), (6, 8)))
    assert not KymoLine(t, y, blank_kymo, "red").in_rect(((3, 6), (4, 8)))
    assert KymoLine(t, y, blank_kymo, "red").in_rect(((3, 6), (5, 8)))

    assert KymoLine([2], [6], blank_kymo, "red").in_rect(((2, 5), (3, 8)))
    assert KymoLine([2], [5], blank_kymo, "red").in_rect(((2, 5), (3, 8)))
    assert not KymoLine([4], [6], blank_kymo, "red").in_rect(((2, 5), (3, 8)))
    assert not KymoLine([1], [6], blank_kymo, "red").in_rect(((2, 5), (3, 8)))
    assert not KymoLine([2], [4], blank_kymo, "red").in_rect(((2, 5), (3, 8)))
    assert not KymoLine([2], [8], blank_kymo, "red").in_rect(((2, 5), (3, 8)))


def test_kymoline_selection_non_unit_calibration():
    kymo = generate_kymo(
        "",
        np.ones((1, 3)),
        pixel_size_nm=5000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0
    )

    time, pos = np.array([4, 5, 6]), np.array([7, 7, 7])
    assert not KymoLine(time, pos, kymo, "red").in_rect(((4, 6 * 5), (6, 7 * 5)))
    assert KymoLine(time, pos, kymo, "red").in_rect(((4, 6 * 5), (6, 8 * 5)))


def test_kymolines_removal(blank_kymo):
    k1 = KymoLine(np.array([1, 2, 3]), np.array([1, 1, 1]), blank_kymo, "red")
    k2 = KymoLine(np.array([2, 3, 4]), np.array([2, 2, 2]), blank_kymo, "red")
    k3 = KymoLine(np.array([3, 4, 5]), np.array([3, 3, 3]), blank_kymo, "red")

    def verify(rect, resulting_lines):
        k = KymoLineGroup([k1, k2, k3])
        k.remove_lines_in_rect(rect)
        assert len(k._src) == len(resulting_lines)
        assert all([l1 == l2 for l1, l2 in zip(k._src, resulting_lines)])

    verify([[5, 3], [6, 4]], [k1, k2])
    verify([[6, 3], [5, 4]], [k1, k2])
    verify([[6, 5], [5, 3]], [k1, k2])
    verify([[0, 0], [5, 5]], [])
    verify([[15, 3], [16, 4]], [k1, k2, k3])


def test_kymolinegroup(blank_kymo):
    k1 = KymoLine(np.array([1, 2, 3]), np.array([2, 3, 4]), blank_kymo, "red")
    k2 = KymoLine(np.array([2, 3, 4]), np.array([3, 4, 5]), blank_kymo, "red")
    k3 = KymoLine(np.array([3, 4, 5]), np.array([4, 5, 6]), blank_kymo, "red")
    k4 = KymoLine(np.array([4, 5, 6]), np.array([5, 6, 7]), blank_kymo, "red")

    lines = KymoLineGroup([k1, k2, k3, k4])
    assert [k for k in lines] == [k1, k2, k3, k4]
    assert len(lines) == 4
    assert lines[0] == k1
    assert lines[1] == k2
    assert lines[0:2][0] == k1
    assert lines[0:2][1] == k2

    with pytest.raises(IndexError):
        lines[0:2][2]

    with pytest.raises(NotImplementedError):
        lines[1] = 4

    lines = KymoLineGroup([k1, k2])
    lines.extend(KymoLineGroup([k3, k4]))
    assert [k for k in lines] == [k1, k2, k3, k4]

    lines = KymoLineGroup([k1, k2, k3])
    lines.extend(k4)
    assert [k for k in lines] == [k1, k2, k3, k4]

    with pytest.raises(TypeError):
        lines.extend(5)


def test_kymoline_concat(blank_kymo):
    k1 = KymoLine(np.array([1, 2, 3]), np.array([1, 1, 1]), blank_kymo, "red")
    k2 = KymoLine(np.array([6, 7, 8]), np.array([2, 2, 2]), blank_kymo, "red")
    k3 = KymoLine(np.array([3, 4, 5]), np.array([3, 3, 3]), blank_kymo, "red")
    k4 = KymoLine(np.array([8, 9, 10]), np.array([3, 3, 3]), blank_kymo, "red")
    group = KymoLineGroup([k1, k2, k3])

    # Test whether overlapping time raises.
    with pytest.raises(RuntimeError):
        group._concatenate_lines(k1, k3)

    # Kymolines have to be added sequentially. Check whether the wrong order raises.
    with pytest.raises(RuntimeError):
        group._concatenate_lines(k2, k1)

    # Check whether a kymoline that's not in the group raises (should only be able to merge
    # within group)
    with pytest.raises(RuntimeError):
        group._concatenate_lines(k1, k4)

    # Check whether a kymoline that's not in the group raises (should only be able to merge
    # within group)
    with pytest.raises(RuntimeError):
        group._concatenate_lines(k4, k1)

    # Check whether an invalid type raises correctly
    with pytest.raises(RuntimeError):
        group._concatenate_lines(k1, 5)

    # Finally do a correct merge
    group._concatenate_lines(k1, k2)
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

    # What we do is we apply the scaling to the indices used in the KymoLine construction. If we
    # define the calibration as exactly the opposite of this, we should get no change.
    kymo = generate_kymo(
        "",
        np.ones((1, 3)),
        pixel_size_nm=1000 / position_scale,
        start=np.int64(20e9),
        dt=np.int64(1e9 / time_scale),
        samples_per_pixel=1,
        line_padding=0
    )
    k = KymoLine(time_scale * time_idx, position_scale * position_idx, kymo, "red")

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

    # What we do is we apply the scaling to the indices used in the KymoLine construction. If we
    # define the calibration as exactly the opposite of this, we should get no change.
    kymo = generate_kymo(
        "",
        np.ones((1, 3)),
        pixel_size_nm=1000 * pixel_size,
        start=np.int64(20e9),
        dt=np.int64(time_step),
        samples_per_pixel=1,
        line_padding=0
    )
    k = KymoLine(time_idx, coordinate / pixel_size, kymo, "red")

    np.testing.assert_allclose(k.estimate_diffusion_ols(max_lag=max_lag), diffusion_const)
    np.testing.assert_allclose(k.estimate_diffusion("ols", max_lag=max_lag).value, diffusion_const)


@pytest.mark.parametrize(
    "max_lag, x_data, y_data",
    [
        (None, [2, 4], (3 * np.arange(1, 3)) ** 2),
        (100, [2, 4, 6, 8], (3 * np.arange(1, 5)) ** 2),
        (3, [2, 4, 6], (3 * np.arange(1, 4)) ** 2),
    ],
)
@cleanup
def test_kymoline_msd_plot(max_lag, x_data, y_data):
    # See whether the plot spins up
    kymo = generate_kymo(
        "",
        np.ones((1, 3)),
        pixel_size_nm=3000,
        start=np.int64(20e9),
        dt=np.int64(2e9),
        samples_per_pixel=1,
        line_padding=0
    )

    plt.figure()
    k = KymoLine(np.arange(1, 6), np.arange(1, 6), kymo, "red")
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
        line_padding=0
    )

    k1 = KymoLine(np.array([1, 2, 3]), np.array([2.5, 3.5, 4.5]), kymo, "red")
    k2 = KymoLine(np.array([2, 3, 4]), np.array([3.5, 4.5, 5.5]), kymo, "red")
    k3 = KymoLine(np.array([3, 4, 5]), np.array([4.5, 5.5, 6.5]), kymo, "red")
    k4 = KymoLine(np.array([4, 5, 6]), np.array([5.5, 6.5, 7.5]), kymo, "red")

    lines = KymoLineGroup([k1, k2, k3, k4])

    # Counting only the first position of each track with the default number of bins
    counts, edges = lines._histogram_binding_events("binding")
    np.testing.assert_equal(counts, [0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(edges, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Counting all points of each track with the default number of bins
    counts, edges = lines._histogram_binding_events("all")
    np.testing.assert_equal(counts, [0, 0, 1, 2, 3, 3, 2, 1, 0, 0])
    np.testing.assert_allclose(edges, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Counting only the first position of each track with custom bin edges
    counts, edges = lines._histogram_binding_events("binding", bins=[2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_equal(counts, [1, 1, 1, 1, 0, 0])
    np.testing.assert_allclose(edges, [2, 3, 4, 5, 6, 7, 8])

    # Counting all points of each track with custom bin edges
    counts, edges = lines._histogram_binding_events("all", bins=[2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_equal(counts, [1, 2, 3, 3, 2, 1])
    np.testing.assert_allclose(edges, [2, 3, 4, 5, 6, 7, 8])


def test_kymolinegroup_copy(blank_kymo):
    k1 = KymoLine(np.array([1, 2, 3]), np.array([1, 1, 1]), blank_kymo, "red")
    k2 = KymoLine(np.array([6, 7, 8]), np.array([2, 2, 2]), blank_kymo, "red")
    group = KymoLineGroup([k1, k2])
    assert id(group._src) != id(copy(group)._src)


def test_kymoline_concat_gaussians(blank_kymo):
    k1 = KymoLine(np.array([1, 2, 3]), np.array([1, 1, 1]), blank_kymo, "red")
    k2 = KymoLine(
        np.array([6, 7, 8]),
        GaussianLocalizationModel(
            np.full(3, 2), np.full(3, 7), np.full(3, 0.5), np.ones(3), np.full(3, False)
        ),
        blank_kymo,
        "red"
    )
    group = KymoLineGroup([k1, k2])

    assert isinstance(k1._localization, LocalizationModel)
    assert isinstance(k2._localization, GaussianLocalizationModel)

    # test concatenation clears gaussian parameters
    group._concatenate_lines(k1, k2)
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
        line_padding=0
    )

    k1 = KymoLine(np.array([1, 2, 3]), np.array([2, 3, 4]), kymo, "red")
    k2 = KymoLine(np.array([2, 3, 4]), np.array([3, 4, 5]), kymo, "red")
    k3 = KymoLine(np.array([3, 4, 5]), np.array([4, 5, 6]), kymo, "red")
    k4 = KymoLine(np.array([4, 5, 6]), np.array([5, 6, 7]), kymo, "red")

    lines = KymoLineGroup([k1, k2, k3, k4])
    x, densities = lines._histogram_binding_profile(3, 0.2, 4)

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
    x, densities = lines._histogram_binding_profile(10, 0.2, 4)
    for j, d in enumerate(densities):
        if j in (0, 7, 8, 9):
            np.testing.assert_equal(d, 0)

    # test no spatial bins
    with pytest.raises(ValueError, match="Number of spatial bins must be >= 2."):
        lines._histogram_binding_profile(11, 0.2, 0)

    # test more bins than frames
    with pytest.raises(ValueError, match="Number of time bins must be <= number of frames."):
        lines._histogram_binding_profile(11, 0.2, 4)

    # no bins requested
    with pytest.raises(ValueError, match="Number of time bins must be > 0."):
        lines._histogram_binding_profile(0, 0.2, 4)


def test_fit_binding_times(blank_kymo):
    k1 = KymoLine(np.array([0, 1, 2]), np.zeros(3), blank_kymo, "red")
    k2 = KymoLine(np.array([2, 3, 4, 5, 6]), np.zeros(5), blank_kymo, "red")
    k3 = KymoLine(np.array([3, 4, 5]), np.zeros(3), blank_kymo, "red")
    k4 = KymoLine(np.array([8, 9]), np.zeros(2), blank_kymo, "red")

    lines = KymoLineGroup([k1, k2, k3, k4])

    dwells = lines.fit_binding_times(1)
    np.testing.assert_allclose(dwells.lifetimes, [1.002547])

    dwells = lines.fit_binding_times(1, exclude_ambiguous_dwells=False)
    np.testing.assert_allclose(dwells.lifetimes, [1.25710457])
