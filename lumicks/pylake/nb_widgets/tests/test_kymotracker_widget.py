import re

import numpy as np
import pytest

from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.kymotracker.kymotrack import KymoTrack, KymoTrackGroup
from lumicks.pylake.nb_widgets.kymotracker_widgets import (
    KymoWidgetGreedy,
    KymotrackerParameter,
    _get_default_parameters,
)


class MockLabel:
    def __init__(self):
        self.value = ""


def calibrate_to_kymo(kymo):
    return (
        lambda coord_idx: kymo.pixelsize_um[0] * coord_idx,
        lambda time_idx: kymo.line_time_seconds * time_idx,
    )


def test_widget_open(kymograph):
    KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)


def test_parameters_kymo(kymograph):
    """Test whether the parameter setting is passed correctly to the algorithm. By setting the threshold to different
    values we can check which tracks are detected and use that to verify that the parameter is used.
    """
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

    # Use remove, but don't actually remove the track because it is only partially inside the
    # rectangle.
    kymo_widget._adding = False
    kymo_widget._track_kymo(*region_select(in_s(12), in_um(8), in_s(15), in_um(9)))
    assert len(kymo_widget.tracks) == 3

    # Remove a single track
    kymo_widget._adding = False
    kymo_widget._track_kymo(*region_select(in_s(10), in_um(8), in_s(15), in_um(9)))
    assert len(kymo_widget.tracks) == 2


def test_save_load_from_ui(kymograph, tmpdir_factory):
    """Check if a round trip through the UI saving function works."""
    testfile = f"{tmpdir_factory.mktemp('pylake')}/kymo.csv"

    kymo_widget = KymoWidgetGreedy(
        kymograph, "red", axis_aspect_ratio=1, use_widgets=False, correct_origin=False
    )
    kymo_widget._labels = {"status": MockLabel(), "warning": MockLabel()}
    kymo_widget._output_filename = testfile
    kymo_widget._save_from_ui()
    assert len(kymo_widget._labels["warning"].value) == 0

    kymo_widget = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)
    kymo_widget._labels = {"status": MockLabel(), "warning": MockLabel()}
    kymo_widget._algorithm_parameters["pixel_threshold"].value = 4
    kymo_widget._track_all()
    kymo_widget._output_filename = testfile

    with pytest.warns(
        RuntimeWarning,
        match=re.escape(
            "Prior to version 1.1.0 the method `sample_from_image` had a bug that assumed "
            "the origin of a pixel to be at the edge rather than the center of the pixel. "
            "Consequently, the sampled window could frequently be off by one pixel. To get "
            "the correct behavior and silence this warning, specify `correct_origin=True` "
            "when opening the kymotracking widget. The old (incorrect) behavior is "
            "maintained until the next major release to ensure backward compatibility. "
            "To silence this warning use `correct_origin=False`."
        ),
    ):
        kymo_widget._save_from_ui()

    assert kymo_widget._labels["warning"].value == (
        "<font color='red'>Sampled intensities are using the wrong pixel origin. To correct "
        "this, add extra argument correct_origin=True when opening the widget. Run "
        "help(lk.KymoWidgetGreedy) for more info."
    )

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
        kymograph.line_time_seconds,
    )
    k2 = KymoTrack(
        np.array([6, 7, 8]),
        np.array([3, 3, 3]),
        kymograph,
        "red",
        kymograph.line_time_seconds,
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
        kymograph.line_time_seconds,
    )
    k2 = KymoTrack(
        np.array([6, 7, 8]),
        np.array([3, 3, 3]),
        kymograph,
        "red",
        kymograph.line_time_seconds,
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
    # The lines of interest here are located at coordinate pixel 12 and 15. The line at 12 will
    # start to incorporate contributions from the line at 15 if the window size is bigger than 5
    # pixels, since a window of 7 leads to a half kernel size of n = 3 which is the lowest window
    # size to incorporate this contribution.
    kymo_widget = KymoWidgetGreedy(
        kymograph,
        "red",
        axis_aspect_ratio=1,
        track_width=5 * kymograph.pixelsize[0] + 0.0001,
        use_widgets=False,
    )
    in_um, in_s = calibrate_to_kymo(kymograph)

    kymo_widget._algorithm_parameters["pixel_threshold"].value = 4
    kymo_widget._track_kymo(*region_select(in_s(5), in_um(12), in_s(20), in_um(13)))
    kymo_widget._refine()

    # With this track_width we'll include the two dim pixels in the test data
    true_coordinate = [12.299547] * 15
    true_coordinate[2] = 12
    true_coordinate[3] = 12
    np.testing.assert_allclose(kymo_widget.tracks[0].coordinate_idx, true_coordinate)


def test_widget_with_calibration(kymograph):
    widget = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)
    np.testing.assert_allclose(
        widget._algorithm_parameters["track_width"].value, kymograph.pixelsize[0] * 4
    )
    np.testing.assert_allclose(widget._algorithm_parameters["track_width"].value, 1.6)
    assert widget._axes.get_ylabel() == r"position (μm)"

    kymo_bp = kymograph.calibrate_to_kbp(10.000)
    widget = KymoWidgetGreedy(kymo_bp, "red", axis_aspect_ratio=1, use_widgets=False)
    np.testing.assert_allclose(
        widget._algorithm_parameters["track_width"].value, kymo_bp.pixelsize[0] * 4
    )
    np.testing.assert_allclose(widget._algorithm_parameters["track_width"].value, 2.0)
    assert widget._axes.get_ylabel() == "position (kbp)"


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
            KymotrackerParameter("p", "d", "int", 5, *rng, True, "yes")


def test_keyword_args(kymograph):
    """Test that only 2 positional arguments can be used."""
    with pytest.raises(TypeError):
        KymoWidgetGreedy(kymograph, "red", 1, use_widgets=False)


@pytest.mark.parametrize(
    "gain,line_time,pixel_size,ref_values",
    (
        # fmt:off
        (1, 2.0, 5.0, {"pixel_threshold": (97, 1, 99, None), "track_width": (4 * 5, 3 * 5, 15 * 5, "μm"), "sigma": (2 * 5, 1 * 5, 5 * 5, "μm"), "velocity": (0, -5 * 5/2, 5 * 5/2, "μm/s")}),
        (0, 2.0, 5.0, {"pixel_threshold": (1, 1, 2, None), "track_width": (4 * 5, 3 * 5, 15 * 5, "μm"), "sigma": (2 * 5, 1 * 5, 5 * 5, "μm"), "velocity": (0, -5 * 5/2, 5 * 5/2, "μm/s")}),
        (1, 4.0, 4.0, {"pixel_threshold": (97, 1, 99, None), "track_width": (4 * 4, 3 * 4, 15 * 4, "μm"), "sigma": (2 * 4, 1 * 4, 5 * 4, "μm"), "velocity": (0, -5 * 4/4, 5 * 4/4, "μm/s")}),
        # fmt:on
    ),
)
def test_default_params_img_dependent(gain, line_time, pixel_size, ref_values):
    kymo = _kymo_from_array(
        gain * np.tile(np.arange(100), (2, 1)),
        "r",
        line_time_seconds=line_time,
        start=100,
        pixel_size_um=pixel_size,
        name="test_kymo",
    )

    default_params = _get_default_parameters(kymo, "red")
    for key, param in ref_values.items():
        value, mini, maxi, display_unit = param
        np.testing.assert_allclose(default_params[key].value, value, err_msg=f"{key}")
        np.testing.assert_allclose(default_params[key].lower_bound, mini, err_msg=f"{key}")
        np.testing.assert_allclose(default_params[key].upper_bound, maxi, err_msg=f"{key}")
        assert default_params[key].display_unit == display_unit, f"{key}"


def test_split(kymograph, mockevent):
    kymo_widget = KymoWidgetGreedy(kymograph, "red", axis_aspect_ratio=1, use_widgets=False)
    kymo_widget._labels = {"warning": MockLabel(), "status": MockLabel()}

    k1 = KymoTrack(
        np.array([0, 1, 2, 3, 6, 7, 8, 9, 10]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        kymograph,
        "red",
        kymograph.line_time_seconds,
    )
    kymo_widget.tracks = KymoTrackGroup([k1])

    # Go into track connection mode
    kymo_widget._select_state({"new": "Split Tracks"})

    assert len(kymo_widget.tracks) == 1

    in_um, in_s = calibrate_to_kymo(kymograph)

    # Click too far from the line to split
    kymo_widget._track_splitter.button_down(mockevent(kymo_widget._axes, in_s(3), in_um(5), 3, 0))
    assert len(kymo_widget.tracks) == 1

    # Split line
    kymo_widget._track_splitter.button_down(mockevent(kymo_widget._axes, in_s(3), in_um(1), 3, 0))
    assert len(kymo_widget.tracks) == 2

    kymo_widget._track_splitter.button_down(mockevent(kymo_widget._axes, in_s(7), in_um(1), 3, 0))
    assert len(kymo_widget.tracks) == 2
    assert kymo_widget._labels["warning"].value == (
        "<font color='red'>One track was below the minimum length threshold and was filtered. "
        "Decrease the minimum length if this was not intended."
    )

    # Split line where both lines are dropped
    kymo_widget._labels = {"warning": MockLabel()}
    kymo_widget._track_splitter.button_down(mockevent(kymo_widget._axes, in_s(9), in_um(1), 3, 0))
    assert len(kymo_widget.tracks) == 1
    assert kymo_widget._labels["warning"].value == (
        "<font color='red'>Two tracks were below the minimum length threshold and were filtered. "
        "Decrease the minimum length if this was not intended."
    )
