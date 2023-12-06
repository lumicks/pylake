import json

import numpy as np
import pytest
import matplotlib.pyplot as plt

from lumicks.pylake import ImageStack
from lumicks.pylake.detail.widefield import TiffStack
from lumicks.pylake.nb_widgets.image_editing import KymoEditorWidget, ImageEditorWidget
from lumicks.pylake.tests.data.mock_widefield import MockTiffFile, make_alignment_image_data


def make_mock_stack():
    spot_coordinates = ((25, 25), (50, 50))
    warp_parameters = {
        "red_warp_parameters": {"Tx": 0, "Ty": 0, "theta": 0},
        "blue_warp_parameters": {"Tx": 0, "Ty": 0, "theta": 0},
    }
    _, image, description, bit_depth = make_alignment_image_data(
        spot_coordinates, version=2, bit_depth=16, camera="wt", **warp_parameters
    )

    tiff = TiffStack(
        [
            MockTiffFile(
                data=[image] * 3,
                times=[["10", "20", 18], ["20", "30", 28], ["30", "40", 38]],
                description=description,
                bit_depth=bit_depth,
            )
        ],
        align_requested=True,
    )

    return ImageStack.from_dataset(tiff)


def test_editor_scroll(mockevent):
    # start with widget at beginning of stack
    stack = make_mock_stack()
    w = ImageEditorWidget(stack)
    ax = plt.gca()
    assert ax.current_frame == 0

    # next frame
    event = mockevent(ax, 50, 50, "up", False)
    ax.handle_scroll_event(event)
    assert ax.current_frame == 1

    # previous frame
    event = mockevent(ax, 50, 50, "down", False)
    ax.handle_scroll_event(event)
    assert ax.current_frame == 0

    # can't go below first frame
    ax.handle_scroll_event(event)
    assert ax.current_frame == 0

    # start with widget at end of stack
    stack = make_mock_stack()
    w = ImageEditorWidget(stack, frame=2)
    ax = plt.gca()
    assert ax.current_frame == 2

    # can't go past last frame
    event = mockevent(ax, 50, 50, "up", False)
    ax.handle_scroll_event(event)
    assert ax.current_frame == 2


def test_editor_clicks(mockevent):
    stack = make_mock_stack()
    w = ImageEditorWidget(stack)
    ax = plt.gca()

    # test returned stack is same as currently plotted stack
    assert id(w.image) == id(ax._current_image)
    # returned stack is the original stack
    assert id(w.image) == id(stack)
    # no points currently defined
    assert len(ax.current_points) == 0

    # click first tether point
    event = mockevent(ax, 50, 50, 1, False)
    ax.handle_button_event(event)
    assert id(w.image) == id(ax._current_image)  # widget synced with axes
    assert len(ax.current_points) == 1  # one click registered

    # click second tether point
    event = mockevent(ax, 50, 75, 1, False)
    ax.handle_button_event(event)
    assert id(w.image) == id(ax._current_image)  # widget synced with axes
    assert id(w.image) != id(stack)  # stack was updated
    assert len(ax.current_points) == 0  # tether defined, refresh points list


def test_cropping_clicks(region_select):
    stack = make_mock_stack()
    w = ImageEditorWidget(stack)
    ax = plt.gca()

    events = region_select(50, 25, 150, 75)
    ax.handle_crop(*events)
    np.testing.assert_equal(ax.roi_limits, (50, 150, 25, 75))
    np.testing.assert_equal(w.image._src._shape, (50, 100))


def test_kymo_cropping_clicks(kymograph, region_select):
    # without calibration
    w = KymoEditorWidget(kymograph, "red")
    ax = plt.gca()

    events = region_select(20, 1, 80, 7)
    ax.handle_crop(*events)
    np.testing.assert_equal(ax.time_limits, (20, 80))
    np.testing.assert_equal(ax.position_limits, (1, 7))

    new_kymo = w.kymo
    assert new_kymo._calibration.unit == "um"
    np.testing.assert_equal(new_kymo.get_image("red").shape, (16, 12))

    # with calibration
    w = KymoEditorWidget(kymograph, "red", tether_length_kbp=0.3)
    ax = plt.gca()

    events = region_select(20, 1, 80, 7)
    ax.handle_crop(*events)
    np.testing.assert_equal(ax.time_limits, (20, 80))
    np.testing.assert_equal(ax.position_limits, (1, 7))

    new_kymo = w.kymo
    assert new_kymo._calibration.unit == "kbp"
    np.testing.assert_equal(new_kymo.get_image("red").shape, (16, 12))
