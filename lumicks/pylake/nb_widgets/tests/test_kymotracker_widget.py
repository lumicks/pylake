from lumicks.pylake.kymotracker.detail.calibrated_images import CalibratedKymographChannel
from lumicks.pylake.nb_widgets.kymotracker_widgets import KymoWidgetGreedy
from lumicks.pylake.kymotracker.kymoline import KymoLine, KymoLineGroup
from matplotlib.testing.decorators import cleanup
import numpy as np
import pytest


def calibrate_to_kymo(kymo):
    return (
        lambda coord_idx: kymo.pixelsize_um[0] * coord_idx,
        lambda time_idx: kymo.line_time_seconds * time_idx,
    )


@cleanup
def test_widget_open(kymograph):
    KymoWidgetGreedy(kymograph, "red", 1, use_widgets=False)


@cleanup
def test_parameters_kymo(kymograph):
    """Test whether the parameter setting is passed correctly to the algorithm. By setting the threshold to different
    values we can check which lines are detected and use that to verify that the parameter is used."""
    kymo_widget = KymoWidgetGreedy(kymograph, "red", 1, use_widgets=False)
    kymo_widget.algorithm_parameters["pixel_threshold"] = 30
    kymo_widget.track_all()
    assert len(kymo_widget.lines) == 0

    kymo_widget.algorithm_parameters["pixel_threshold"] = 7
    kymo_widget.track_all()
    assert len(kymo_widget.lines) == 1

    kymo_widget.algorithm_parameters["pixel_threshold"] = 2
    kymo_widget.track_all()
    assert len(kymo_widget.lines) == 3


@cleanup
def test_invalid_algorithm_parameter(kymograph):
    kymo_widget = KymoWidgetGreedy(kymograph, "red", 1, use_widgets=False)
    with pytest.raises(TypeError):
        kymo_widget.algorithm_parameters["bob"] = 5
        kymo_widget.track_all()


@cleanup
def test_track_kymo(kymograph, region_select):
    kymo_widget = KymoWidgetGreedy(kymograph, "red", 1, use_widgets=False)
    assert len(kymo_widget.lines) == 0

    in_um, in_s = calibrate_to_kymo(kymograph)

    # Track a line in a particular region. Only a single line exists in this region.
    kymo_widget.algorithm_parameters["pixel_threshold"] = 4
    kymo_widget.track_kymo(*region_select(in_um(8), in_s(10), in_um(9), in_s(20)))
    np.testing.assert_allclose(kymo_widget.lines[0].time_idx, np.arange(10, 20))
    np.testing.assert_allclose(kymo_widget.lines[0].coordinate_idx, [8] * 10)
    assert len(kymo_widget.lines) == 1

    # Verify that if we track the same region, the old one gets deleted and we track the same line
    # again.
    kymo_widget.track_kymo(*region_select(in_um(8), in_s(15), in_um(9), in_s(20)))
    np.testing.assert_allclose(kymo_widget.lines[0].time_idx, np.arange(15, 20))
    np.testing.assert_allclose(kymo_widget.lines[0].coordinate_idx, [8] * 5)
    assert len(kymo_widget.lines) == 1

    # Tracking all lines will result in all lines being found.
    kymo_widget.track_all()
    assert len(kymo_widget.lines) == 3

    # Remove a single line
    kymo_widget.adding = False
    kymo_widget.track_kymo(*region_select(in_um(8), in_s(15), in_um(9), in_s(20)))
    assert len(kymo_widget.lines) == 2


def test_save_load_from_ui(kymograph, tmpdir_factory):
    """Check if a round trip through the UI saving function works."""
    testfile = f"{tmpdir_factory.mktemp('pylake')}/kymo.csv"

    kymo_widget = KymoWidgetGreedy(kymograph, "red", 1, use_widgets=False)
    kymo_widget.algorithm_parameters["pixel_threshold"] = 4
    kymo_widget.track_all()
    kymo_widget.output_filename = testfile
    kymo_widget._save_from_ui()

    lines = kymo_widget.lines

    kymo_widget = KymoWidgetGreedy(kymograph, "red", 1, use_widgets=False)
    assert len(kymo_widget.lines) == 0

    kymo_widget.output_filename = testfile
    kymo_widget._load_from_ui()

    for l1, l2 in zip(lines, kymo_widget.lines):
        np.testing.assert_allclose(l1.time_idx, l2.time_idx)
        np.testing.assert_allclose(l1.coordinate_idx, l2.coordinate_idx)


def test_refine_from_widget(kymograph, region_select):
    kymo_widget = KymoWidgetGreedy(kymograph, "red", 1, use_widgets=False)
    in_um, in_s = calibrate_to_kymo(kymograph)

    # Test whether error is handled when refining before tracking / loading
    class MockLabel:
        def __init__(self):
            self.value = ""

    kymo_widget._label = MockLabel()
    kymo_widget.refine()
    assert kymo_widget._label.value == "You need to track or load kymograph lines before you can " \
                                       "refine them"

    kymo_widget.algorithm_parameters["pixel_threshold"] = 4
    kymo_widget.track_kymo(*region_select(in_um(12), in_s(5), in_um(13), in_s(20)))
    np.testing.assert_allclose(kymo_widget.lines[0].time_idx, np.hstack(([5, 6], np.arange(9, 20))))
    np.testing.assert_allclose(kymo_widget.lines[0].coordinate_idx, [12] * 13)
    assert len(kymo_widget.lines) == 1

    kymo_widget.refine()
    np.testing.assert_allclose(kymo_widget.lines[0].time_idx, np.arange(5, 20))
    np.testing.assert_allclose(kymo_widget.lines[0].coordinate_idx, [12] * 15)
    assert len(kymo_widget.lines) == 1


def test_stitch(kymograph, mockevent):
    kymo_widget = KymoWidgetGreedy(kymograph, "red", 1, use_widgets=False)

    k1 = KymoLine(
        np.array([1, 2, 3]),
        np.array([1, 1, 1]),
        CalibratedKymographChannel.from_kymo(kymograph, "red"),
    )
    k2 = KymoLine(
        np.array([6, 7, 8]),
        np.array([3, 3, 3]),
        CalibratedKymographChannel.from_kymo(kymograph, "red"),
    )
    kymo_widget.lines = KymoLineGroup([k1, k2])

    # Go into line connection mode
    kymo_widget._select_state({"new": "Connect Lines"})

    assert len(kymo_widget.lines) == 2

    in_um, in_s = calibrate_to_kymo(kymograph)

    # Drag but stop too early (not leading to a connected line)
    kymo_widget._line_connector.button_down(mockevent(kymo_widget._axes, in_s(3), in_um(1), 3, 0))
    kymo_widget._line_connector.button_release(
        mockevent(kymo_widget._axes, in_s(4), in_um(3), 3, 0)
    )
    assert len(kymo_widget.lines) == 2

    # Drag all the way (stitch the two)
    kymo_widget._line_connector.button_down(mockevent(kymo_widget._axes, in_s(3), in_um(1), 3, 0))
    kymo_widget._line_connector.button_release(
        mockevent(kymo_widget._axes, in_s(6), in_um(3), 3, 0)
    )

    # Verify the stitched line
    np.testing.assert_allclose(kymo_widget.lines[0].time_idx, [1, 2, 3, 6, 7, 8])
    np.testing.assert_allclose(kymo_widget.lines[0].coordinate_idx, [1, 1, 1, 3, 3, 3])
    assert len(kymo_widget.lines) == 1


def test_refine_line_width_units(kymograph, region_select):
    kymo_widget = KymoWidgetGreedy(kymograph, "red", 1, line_width=2, use_widgets=False)
    in_um, in_s = calibrate_to_kymo(kymograph)

    kymo_widget.algorithm_parameters["pixel_threshold"] = 4
    kymo_widget.track_kymo(*region_select(in_um(12), in_s(5), in_um(13), in_s(20)))
    kymo_widget.refine()

    # With this line_width we'll include the two dim pixels in the test data
    true_coordinate = [12.176471] * 15
    true_coordinate[2] = 12
    true_coordinate[3] = 12
    np.testing.assert_allclose(kymo_widget.lines[0].coordinate_idx, true_coordinate)
