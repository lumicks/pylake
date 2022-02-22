import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup
from lumicks.pylake.kymotracker.kymoline import *
from lumicks.pylake.kymotracker.detail.trace_line_2d import KymoLineData


def test_kymo_line():
    channel = CalibratedKymographChannel("test_data", np.array([[]]), 1e9, 1)

    k1 = KymoLine(np.array([1, 2, 3]), np.array([2, 3, 4]), channel)
    np.testing.assert_allclose(k1[1], [2, 3])
    np.testing.assert_allclose(k1[-1], [3, 4])
    np.testing.assert_allclose(k1[0:2], [[1, 2], [2, 3]])
    np.testing.assert_allclose(k1[0:2][:, 1], [2, 3])

    k2 = KymoLine(np.array([4, 5, 6]), np.array([5, 6, 7]), channel)
    np.testing.assert_allclose((k1 + k2)[:], [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    np.testing.assert_allclose(k1.extrapolate(True, 3, 2.0), [5, 6])

    # Need at least 2 points for linear extrapolation
    with pytest.raises(AssertionError):
        KymoLine([1], [1], channel).extrapolate(True, 5, 2.0)

    with pytest.raises(AssertionError):
        KymoLine([1, 2, 3], [1, 2, 3], channel).extrapolate(True, 1, 2.0)

    k1 = KymoLine([1, 2, 3], [1, 2, 3], channel)
    k2 = k1.with_offset(2, 2)
    assert id(k2) != id(k1)

    np.testing.assert_allclose(k2.coordinate_idx, [3, 4, 5])
    assert k1._image == channel
    assert k2._image == channel


def test_kymoline_selection():
    channel = CalibratedKymographChannel("test_data", np.array([[]]), 1e9, 1)

    t = np.array([4, 5, 6])
    y = np.array([7, 7, 7])
    assert not KymoLine(t, y, channel).in_rect(((4, 6), (6, 7)))
    assert KymoLine(t, y, channel).in_rect(((4, 6), (6, 8)))
    assert not KymoLine(t, y, channel).in_rect(((3, 6), (4, 8)))
    assert KymoLine(t, y, channel).in_rect(((3, 6), (5, 8)))

    assert KymoLine([2], [6], channel).in_rect(((2, 5), (3, 8)))
    assert KymoLine([2], [5], channel).in_rect(((2, 5), (3, 8)))
    assert not KymoLine([4], [6], channel).in_rect(((2, 5), (3, 8)))
    assert not KymoLine([1], [6], channel).in_rect(((2, 5), (3, 8)))
    assert not KymoLine([2], [4], channel).in_rect(((2, 5), (3, 8)))
    assert not KymoLine([2], [8], channel).in_rect(((2, 5), (3, 8)))


def test_kymoline_selection_non_unit_calibration():
    channel = CalibratedKymographChannel("test_data", np.array([[]]), 1e9, 5)

    time, pos = np.array([4, 5, 6]), np.array([7, 7, 7])
    assert not KymoLine(time, pos, channel).in_rect(((4, 6 * 5), (6, 7 * 5)))
    assert KymoLine(time, pos, channel).in_rect(((4, 6 * 5), (6, 8 * 5)))


def test_kymoline_from_kymolinedata():
    channel = CalibratedKymographChannel("test_data", np.array([[]]), 1e9, 5)
    time, pos = np.array([4, 5, 6]), np.array([7, 7, 7])
    kl = KymoLine._from_kymolinedata(KymoLineData(time, pos), channel)
    np.testing.assert_allclose(kl.time_idx, time)
    np.testing.assert_allclose(kl.coordinate_idx, pos)
    assert id(kl._image) == id(channel)


def test_kymolines_removal():
    channel = CalibratedKymographChannel("test_data", np.array([[]]), 1e9, 1)

    k1 = KymoLine(np.array([1, 2, 3]), np.array([1, 1, 1]), channel)
    k2 = KymoLine(np.array([2, 3, 4]), np.array([2, 2, 2]), channel)
    k3 = KymoLine(np.array([3, 4, 5]), np.array([3, 3, 3]), channel)

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


def test_kymolinegroup():
    channel = CalibratedKymographChannel("test_data", np.array([[]]), 1e9, 1)

    k1 = KymoLine(np.array([1, 2, 3]), np.array([2, 3, 4]), channel)
    k2 = KymoLine(np.array([2, 3, 4]), np.array([3, 4, 5]), channel)
    k3 = KymoLine(np.array([3, 4, 5]), np.array([4, 5, 6]), channel)
    k4 = KymoLine(np.array([4, 5, 6]), np.array([5, 6, 7]), channel)

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


def test_kymoline_concat():
    channel = CalibratedKymographChannel("test_data", np.array([[]]), 1e9, 1)

    k1 = KymoLine(np.array([1, 2, 3]), np.array([1, 1, 1]), channel)
    k2 = KymoLine(np.array([6, 7, 8]), np.array([2, 2, 2]), channel)
    k3 = KymoLine(np.array([3, 4, 5]), np.array([3, 3, 3]), channel)
    k4 = KymoLine(np.array([8, 9, 10]), np.array([3, 3, 3]), channel)
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
    image = CalibratedKymographChannel(
        "test", np.array([[]]), int(1e9 / time_scale), 1.0 / position_scale
    )
    k = KymoLine(time_scale * time_idx, position_scale * position_idx, image=image)

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
    image = CalibratedKymographChannel("test", np.array([[]]), time_step, pixel_size)
    k = KymoLine(time_idx, coordinate / pixel_size, image=image)

    np.testing.assert_allclose(k.estimate_diffusion_ols(max_lag=max_lag), diffusion_const)


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
    image = CalibratedKymographChannel("test", np.array([[]]), int(2e9), 3.0)

    plt.figure()
    k = KymoLine(np.arange(1, 6), np.arange(1, 6), image=image)
    k.plot_msd(max_lag=max_lag)
    np.testing.assert_allclose(plt.gca().lines[0].get_xdata(), x_data)
    np.testing.assert_allclose(plt.gca().lines[0].get_ydata(), y_data)


def test_binding_histograms():
    channel = CalibratedKymographChannel("test_data", np.zeros((10, 10)), 1e9, 1)

    k1 = KymoLine(np.array([1, 2, 3]), np.array([2.5, 3.5, 4.5]), channel)
    k2 = KymoLine(np.array([2, 3, 4]), np.array([3.5, 4.5, 5.5]), channel)
    k3 = KymoLine(np.array([3, 4, 5]), np.array([4.5, 5.5, 6.5]), channel)
    k4 = KymoLine(np.array([4, 5, 6]), np.array([5.5, 6.5, 7.5]), channel)

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


def test_kymolinegroup_copy():
    k1 = KymoLine(np.array([1, 2, 3]), np.array([1, 1, 1]), [])
    k2 = KymoLine(np.array([6, 7, 8]), np.array([2, 2, 2]), [])
    group = KymoLineGroup([k1, k2])
    assert id(group._src) != id(copy(group)._src)


def test_binding_profile_histogram():
    channel = CalibratedKymographChannel("test_data", np.ones((10, 10)), 1e9, 1)

    k1 = KymoLine(np.array([1, 2, 3]), np.array([2, 3, 4]), channel)
    k2 = KymoLine(np.array([2, 3, 4]), np.array([3, 4, 5]), channel)
    k3 = KymoLine(np.array([3, 4, 5]), np.array([4, 5, 6]), channel)
    k4 = KymoLine(np.array([4, 5, 6]), np.array([5, 6, 7]), channel)

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
