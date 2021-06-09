import matplotlib.pyplot as plt
import numpy as np
from lumicks.pylake.nb_widgets.detail.mouse import MouseDragCallback


def open_plot():
    plt.plot([1, 2, 3, 4, 5], [5, 6, 7, 8, 9])
    plt.xlim([0, 5])
    plt.ylim([5, 10])
    return plt.gca()


def test_mouse_drag(mockevent):
    ax = open_plot()
    current_x, current_y, current_dx, current_dy = 0, 0, 0, 0

    def callback(drag_event):
        nonlocal current_x, current_y, current_dx, current_dy
        current_x, current_y, current_dx, current_dy = (
            drag_event.x,
            drag_event.y,
            drag_event.dx,
            drag_event.dy,
        )

    button = 1
    mouse_drag = MouseDragCallback(ax, button, callback)
    assert not mouse_drag.dragging

    mouse_drag.button_down(mockevent(ax, 1, 1, button + 1, 1))  # Wrong button for this callback
    assert not mouse_drag.dragging

    mouse_drag.button_down(mockevent(ax, 1, 1, button, 1))  # Blocked widget
    assert not mouse_drag.dragging

    mouse_drag.button_down(mockevent(ax, 1, 1, button, 0))
    assert mouse_drag.dragging

    mouse_drag.button_down(mockevent(ax, 1, 1, button, 0))
    np.testing.assert_allclose(current_x, 0)

    mouse_drag.handle_motion(mockevent(ax, 2, 2, button, 0))
    np.testing.assert_allclose(current_x, 2)
    np.testing.assert_allclose(current_y, 2)
    np.testing.assert_allclose(current_dx, 1)
    np.testing.assert_allclose(current_dy, 1)

    mouse_drag.handle_motion(mockevent(ax, 5, 5, button, 0))
    np.testing.assert_allclose(current_x, 5)
    np.testing.assert_allclose(current_y, 5)
    np.testing.assert_allclose(current_dx, 3)
    np.testing.assert_allclose(current_dy, 3)

    mouse_drag.button_release(mockevent(ax, 1, 1, button, 0))
    mouse_drag.handle_motion(mockevent(ax, 15, 15, button, 0))
    np.testing.assert_allclose(current_x, 5)
    np.testing.assert_allclose(current_y, 5)
    np.testing.assert_allclose(current_dx, 3)
    np.testing.assert_allclose(current_dy, 3)


def test_set_active():
    ax = open_plot()
    button = 1
    mouse_drag = MouseDragCallback(ax, button, [])

    # Callbacks should be on initially
    for cib in mouse_drag._callback_ids.values():
        assert cib

    # Inactive handler should not be dragging
    mouse_drag.dragging = True
    mouse_drag.set_active(False)
    assert not mouse_drag.dragging

    # Disabling the handler should remove all the callbacks
    for cib in mouse_drag._callback_ids.values():
        assert cib is None

    mouse_drag.set_active(True)

    # Enabling them should put them back
    for cib in mouse_drag._callback_ids.values():
        assert cib


def test_callbacks(mockevent):
    ax = open_plot()
    button = 1
    press = release = drag = False

    def callback_press(drag_event):
        nonlocal press
        press = True

    def callback_release(drag_event):
        nonlocal release
        release = True

    def callback_drag(drag_event):
        nonlocal drag
        drag = True

    mouse_drag = MouseDragCallback(ax, button, callback_drag, callback_press, callback_release)

    mouse_drag.button_down(mockevent(ax, 1, 1, button, 0))
    assert press
    assert not release
    assert not drag


def test_mouse_drag_lim_change(mockevent):
    """When limits change, the coordinate system changes. This test tests whether that is taken into account
    correctly."""
    ax = open_plot()
    current_x, current_y, current_dx, current_dy = 0, 0, 0, 0

    def callback(event):
        nonlocal current_x, current_y, current_dx, current_dy
        xl = plt.gca().get_xlim()
        plt.xlim([xl[0] + event.dx, xl[1] + event.dx])
        current_x, current_y, current_dx, current_dy = event.x, event.y, event.dx, event.dy

    mouse_drag = MouseDragCallback(ax, 1, callback)
    mouse_drag.button_down(mockevent(ax, 1, 1, 1, 0))
    mouse_drag.handle_motion(mockevent(ax, 1, 1, 1, 0))

    mouse_drag.handle_motion(mockevent(ax, 2, 2, 1, 0))
    np.testing.assert_allclose(current_x, 2)
    np.testing.assert_allclose(current_y, 2)
    np.testing.assert_allclose(current_dx, 1)
    np.testing.assert_allclose(current_dy, 1)

    mouse_drag.handle_motion(mockevent(ax, 2, 2, 1, 0))
    np.testing.assert_allclose(current_x, 2)
    np.testing.assert_allclose(current_y, 2)
    np.testing.assert_allclose(current_dx, -1)
    np.testing.assert_allclose(current_dy, 0, atol=1e-15)
