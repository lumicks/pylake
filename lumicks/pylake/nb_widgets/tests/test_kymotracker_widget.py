from lumicks.pylake.kymotracker.detail.calibrated_images import CalibratedKymographChannel
from lumicks.pylake.nb_widgets.kymotracker_widgets import KymoWidgetGreedy
from lumicks.pylake.kymotracker.kymoline import KymoLine, KymoLineGroup
from matplotlib.testing.decorators import cleanup
import numpy as np
import pytest


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

    # Track a line in a particular region. Only a single line exists in this region.
    kymo_widget.algorithm_parameters["pixel_threshold"] = 4
    kymo_widget.track_kymo(*region_select(8, 10, 9, 20))
    np.testing.assert_allclose(kymo_widget.lines[0].time_idx, np.arange(10, 20))
    np.testing.assert_allclose(kymo_widget.lines[0].coordinate_idx, [8] * 10)
    assert len(kymo_widget.lines) == 1

    # Verify that if we track the same region, the old one gets deleted and we track the same line again.
    kymo_widget.track_kymo(*region_select(8, 15, 9, 20))
    np.testing.assert_allclose(kymo_widget.lines[0].time_idx, np.arange(15, 20))
    np.testing.assert_allclose(kymo_widget.lines[0].coordinate_idx, [8] * 5)
    assert len(kymo_widget.lines) == 1

    # Tracking all lines will result in all lines being found.
    kymo_widget.track_all()
    assert len(kymo_widget.lines) == 3

    # Remove a single line
    kymo_widget.adding = False
    kymo_widget.track_kymo(*region_select(8, 15, 9, 20))
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

    kymo_widget.algorithm_parameters["pixel_threshold"] = 4
    kymo_widget.track_kymo(*region_select(12, 5, 13, 20))
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

    # Drag but stop too early (not leading to a connected line)
    kymo_widget._line_connector.button_down(mockevent(kymo_widget._axes, 3, 1, 3, 0))
    kymo_widget._line_connector.button_release(mockevent(kymo_widget._axes, 4, 3, 3, 0))
    assert len(kymo_widget.lines) == 2

    # Drag all the way (stitch the two)
    kymo_widget._line_connector.button_down(mockevent(kymo_widget._axes, 3, 1, 3, 0))
    kymo_widget._line_connector.button_release(mockevent(kymo_widget._axes, 6, 3, 3, 0))

    # Verify the stitched line
    np.testing.assert_allclose(kymo_widget.lines[0].seconds, [1, 2, 3, 6, 7, 8])
    np.testing.assert_allclose(kymo_widget.lines[0].position, [1, 1, 1, 3, 3, 3])
    assert len(kymo_widget.lines) == 1
