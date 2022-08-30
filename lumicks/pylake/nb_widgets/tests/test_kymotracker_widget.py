from lumicks.pylake.nb_widgets.kymotracker_widgets import KymoWidgetGreedy, KymotrackerParameter
from lumicks.pylake.kymotracker.kymotrack import KymoTrack, KymoTrackGroup
from matplotlib.testing.decorators import cleanup
import numpy as np
import re
import pytest


def calibrate_to_kymo(kymo):
    return (
        lambda coord_idx: kymo.pixelsize_um[0] * coord_idx,
        lambda time_idx: kymo.line_time_seconds * time_idx,
    )


@cleanup
def test_widget_open(kymograph):
    KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)


@cleanup
def test_deprecations(kymograph):
    kymo_widget = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)
    with pytest.warns(DeprecationWarning):
        kymo_widget.lines


@cleanup
def test_parameters_kymo(kymograph):
    """Test whether the parameter setting is passed correctly to the algorithm. By setting the threshold to different
    values we can check which tracks are detected and use that to verify that the parameter is used."""
    kymo_widget = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)
    kymo_widget._algorithm_parameters["pixel_threshold"].value = 30
    kymo_widget._track_all()
    assert len(kymo_widget.tracks) == 0

    kymo_widget._algorithm_parameters["pixel_threshold"].value = 7
    kymo_widget._track_all()
    assert len(kymo_widget.tracks) == 1

    kymo_widget._algorithm_parameters["pixel_threshold"].value = 2
    kymo_widget._track_all()
    assert len(kymo_widget.tracks) == 3


@cleanup
def test_invalid_algorithm_parameter(kymograph):
    kymo_widget = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)
    with pytest.raises(KeyError):
        kymo_widget._algorithm_parameters["bob"].value = 5
        kymo_widget.track_all()

    with pytest.raises(AttributeError):
        kymo_widget._algorithm_parameters["bob"] = 5
        kymo_widget.track_all()

    with pytest.raises(TypeError):
        kymo_widget._algorithm_parameters["bob"] = KymotrackerParameter(
            "bob", "nonsense", "int", 5, (1, 10)
        )
        kymo_widget.track_all()


@cleanup
def test_aspect_ratio(kymograph, region_select):
    for requested_aspect in (2, 3, 5):
        kymo_widget = KymoWidgetGreedy(
            kymograph, "red", use_widgets=False, axis_aspect_ratio=requested_aspect
        )
        ax = kymo_widget._axes
        aspect = ax.get_xlim()[1] / ax.get_ylim()[0]
        np.testing.assert_allclose(
            aspect,
            requested_aspect,
            rtol=0.05,
            err_msg=f"aspect ratio = {requested_aspect} failed.",
        )


@cleanup
def test_track_kymo(kymograph, region_select):
    kymo_widget = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)
    assert len(kymo_widget.tracks) == 0

    in_um, in_s = calibrate_to_kymo(kymograph)

    # Track a particular region. Only a single track exists in this region.
    kymo_widget._algorithm_parameters["pixel_threshold"].value = 4
    kymo_widget._track_kymo(*region_select(in_s(10), in_um(8), in_s(20), in_um(9)))
    np.testing.assert_allclose(kymo_widget.tracks[0].time_idx, np.arange(10, 20))
    np.testing.assert_allclose(kymo_widget.tracks[0].coordinate_idx, [8] * 10)
    assert len(kymo_widget.tracks) == 1

    # Verify that if we track the same region, the old one gets deleted and we get the same
    # track again.
    kymo_widget._track_kymo(*region_select(in_s(15), in_um(8), in_s(20), in_um(9)))
    np.testing.assert_allclose(kymo_widget.tracks[0].time_idx, np.arange(15, 20))
    np.testing.assert_allclose(kymo_widget.tracks[0].coordinate_idx, [8] * 5)
    assert len(kymo_widget.tracks) == 1

    # Tracking the full kymo will result in all tracks being found.
    kymo_widget._track_all()
    assert len(kymo_widget.tracks) == 3

    # Remove a single track
    kymo_widget._adding = False
    kymo_widget._track_kymo(*region_select(in_s(15), in_um(8), in_s(20), in_um(9)))
    assert len(kymo_widget.tracks) == 2


def test_save_load_from_ui(kymograph, tmpdir_factory):
    """Check if a round trip through the UI saving function works."""
    testfile = f"{tmpdir_factory.mktemp('pylake')}/kymo.csv"

    kymo_widget = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)
    kymo_widget._algorithm_parameters["pixel_threshold"].value = 4
    kymo_widget._track_all()
    kymo_widget._output_filename = testfile
    kymo_widget._save_from_ui()

    tracks = kymo_widget.tracks

    kymo_widget = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)
    assert len(kymo_widget.tracks) == 0

    kymo_widget._output_filename = testfile
    kymo_widget._load_from_ui()

    for l1, l2 in zip(tracks, kymo_widget.tracks):
        np.testing.assert_allclose(l1.time_idx, l2.time_idx)
        np.testing.assert_allclose(l1.coordinate_idx, l2.coordinate_idx)


def test_refine_from_widget(kymograph, region_select):
    kymo_widget = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)
    in_um, in_s = calibrate_to_kymo(kymograph)

    # Test whether error is handled when refining before tracking / loading
    class MockLabel:
        def __init__(self):
            self.value = ""

    kymo_widget._labels = {"status": MockLabel()}
    kymo_widget._refine()
    assert (
        kymo_widget._labels["status"].value
        == "You need to track this kymograph or load tracks before you can refine them"
    )

    kymo_widget._algorithm_parameters["pixel_threshold"].value = 4
    kymo_widget._track_kymo(*region_select(in_s(5), in_um(12), in_s(20), in_um(13)))
    np.testing.assert_allclose(
        kymo_widget.tracks[0].time_idx, np.hstack(([5, 6], np.arange(9, 20)))
    )
    np.testing.assert_allclose(kymo_widget.tracks[0].coordinate_idx, [12] * 13)
    assert len(kymo_widget.tracks) == 1

    kymo_widget._refine()
    np.testing.assert_allclose(kymo_widget.tracks[0].time_idx, np.arange(5, 20))
    np.testing.assert_allclose(kymo_widget.tracks[0].coordinate_idx, [12] * 15)
    assert len(kymo_widget.tracks) == 1


def test_stitch(kymograph, mockevent):
    kymo_widget = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)

    k1 = KymoTrack(
        np.array([1, 2, 3]),
        np.array([1, 1, 1]),
        kymograph,
        "red",
    )
    k2 = KymoTrack(
        np.array([6, 7, 8]),
        np.array([3, 3, 3]),
        kymograph,
        "red",
    )
    kymo_widget.tracks = KymoTrackGroup([k1, k2])

    # Go into track connection mode
    kymo_widget._select_state({"new": "Connect Tracks"})

    assert len(kymo_widget.tracks) == 2

    in_um, in_s = calibrate_to_kymo(kymograph)

    # Drag but stop too early (not leading to a connected track)
    kymo_widget._track_connector.button_down(mockevent(kymo_widget._axes, in_s(3), in_um(1), 3, 0))
    kymo_widget._track_connector.button_release(
        mockevent(kymo_widget._axes, in_s(4), in_um(3), 3, 0)
    )
    assert len(kymo_widget.tracks) == 2

    # Drag all the way (stitch the two)
    kymo_widget._track_connector.button_down(mockevent(kymo_widget._axes, in_s(3), in_um(1), 3, 0))
    kymo_widget._track_connector.button_release(
        mockevent(kymo_widget._axes, in_s(6), in_um(3), 3, 0)
    )

    # Verify the stitched track
    np.testing.assert_allclose(kymo_widget.tracks[0].time_idx, [1, 2, 3, 6, 7, 8])
    np.testing.assert_allclose(kymo_widget.tracks[0].coordinate_idx, [1, 1, 1, 3, 3, 3])
    assert len(kymo_widget.tracks) == 1


@pytest.mark.parametrize(
    "start,stop,same_track", [(2, 7, False), (3, 7, False), (2, 6, False), (2, 5, True)]
)
def test_stitch_anywhere(start, stop, same_track, kymograph, mockevent):
    kymo_widget = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)

    k1 = KymoTrack(
        np.array([1, 2, 3, 4, 5]),
        np.array([1, 1, 1, 3, 3]),
        kymograph,
        "red",
    )
    k2 = KymoTrack(
        np.array([6, 7, 8]),
        np.array([3, 3, 3]),
        kymograph,
        "red",
    )
    kymo_widget.tracks = KymoTrackGroup([k1, k2])

    # Go into track connection mode
    kymo_widget._select_state({"new": "Connect Tracks"})
    in_um, in_s = calibrate_to_kymo(kymograph)

    # Merge points
    kymo_widget._track_connector.button_down(
        mockevent(kymo_widget._axes, in_s(start), in_um(1), 3, 0)
    )
    kymo_widget._track_connector.button_release(
        mockevent(kymo_widget._axes, in_s(stop), in_um(3), 3, 0)
    )

    # Verify the stitched track
    if not same_track:
        time_result = np.hstack((np.arange(start) + 1, np.arange(stop, 9)))
        coord_result = np.hstack((np.full(start, 1), np.full(9 - stop, 3)))
    else:
        time_result = np.hstack((np.arange(start) + 1, np.arange(stop, 6)))
        coord_result = np.hstack((np.full(start, 1), np.full(6 - stop, 3)))
    np.testing.assert_allclose(kymo_widget.tracks[0].time_idx, time_result)
    np.testing.assert_allclose(kymo_widget.tracks[0].coordinate_idx, coord_result)
    assert len(kymo_widget.tracks) == 2 if same_track else 1


def test_refine_track_width_units(kymograph, region_select):
    kymo_widget = KymoWidgetGreedy(
        kymograph, "red", axis_aspect_ratio=1, track_width=2, use_widgets=False
    )
    in_um, in_s = calibrate_to_kymo(kymograph)

    kymo_widget._algorithm_parameters["pixel_threshold"].value = 4
    kymo_widget._track_kymo(*region_select(in_s(5), in_um(12), in_s(20), in_um(13)))
    kymo_widget._refine()

    # With this track_width we'll include the two dim pixels in the test data
    true_coordinate = [12.176471] * 15
    true_coordinate[2] = 12
    true_coordinate[3] = 12
    np.testing.assert_allclose(kymo_widget.tracks[0].coordinate_idx, true_coordinate)


@cleanup
def test_widget_with_calibration(kymograph):
    widget = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)
    np.testing.assert_allclose(
        widget._algorithm_parameters["track_width"].value, kymograph.pixelsize[0] * 4
    )
    np.testing.assert_allclose(widget._algorithm_parameters["track_width"].value, 1.6)
    assert widget._axes.get_ylabel() == r"position ($\mu$m)"

    kymo_bp = kymograph.calibrate_to_kbp(10.000)
    widget = KymoWidgetGreedy(kymo_bp, "red", axis_aspect_ratio=1, use_widgets=False)
    np.testing.assert_allclose(
        widget._algorithm_parameters["track_width"].value, kymo_bp.pixelsize[0] * 4
    )
    np.testing.assert_allclose(widget._algorithm_parameters["track_width"].value, 2.0)
    assert widget._axes.get_ylabel() == "position (kbp)"


@cleanup
def test_invalid_range_overrides(kymograph):
    with pytest.raises(KeyError, match="Slider range provided for parameter that does not exist"):
        KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, slider_ranges={"wrong_par": (1, 5)})
    with pytest.raises(
        ValueError,
        match="Lower bound should be lower than upper bound for " "parameter pixel_threshold",
    ):
        KymoWidgetGreedy(
            kymograph, "red", axis_aspect_ratio=1, slider_ranges={"pixel_threshold": (5, 1)}
        )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Slider range for parameter pixel_threshold should be given as "
            "(lower bound, upper bound)."
        ),
    ):
        KymoWidgetGreedy(
            kymograph, "red", axis_aspect_ratio=1, slider_ranges={"pixel_threshold": (1, 5, 6)}
        )


@cleanup
def test_valid_override(kymograph):
    """Tests whether the correct ranges make it into the widget data. Unfortunately we cannot test
    the full widget as we cannot spin up the UI."""
    kw = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)
    assert kw._algorithm_parameters["pixel_threshold"].lower_bound == 1
    assert kw._algorithm_parameters["pixel_threshold"].upper_bound == 10

    kw = KymoWidgetGreedy(
        kymograph,
        "red",
        axis_aspect_ratio=1,
        use_widgets=False,
        slider_ranges={"pixel_threshold": (5, 10)},
    )
    assert kw._algorithm_parameters["pixel_threshold"].lower_bound == 5
    assert kw._algorithm_parameters["pixel_threshold"].upper_bound == 10

    kw = KymoWidgetGreedy(
        kymograph,
        "red",
        axis_aspect_ratio=1,
        use_widgets=False,
        slider_ranges={"min_length": (5, 10)},
    )
    assert kw._algorithm_parameters["min_length"].lower_bound == 5
    assert kw._algorithm_parameters["min_length"].upper_bound == 10


def test_valid_default_parameters():
    for rng in ([None, 15], [1, None], [None, None]):
        with pytest.raises(
            ValueError,
            match="Lower and upper bounds must be supplied for widget to be set as visible.",
        ):
            KymotrackerParameter("p", "d", "int", 5, *rng, True)


@cleanup
def test_keyword_args(kymograph):
    """Test that only 2 positional arguments can be used."""
    with pytest.raises(TypeError):
        KymoWidgetGreedy(kymograph, "red", 1, use_widgets=False)
