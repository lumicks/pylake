import matplotlib.pyplot as plt
import numpy as np
from lumicks.pylake.nb_widgets.detail.mouse import MouseDragCallback


def test_mouse_drag(mockevent):
    plt.plot([1, 2, 3, 4, 5], [5, 6, 7, 8, 9])
    plt.xlim([0, 5])
    plt.ylim([5, 10])
    ax = plt.gca()

    current_x, current_y, current_dx, current_dy = 0, 0, 0, 0

    def callback(x, y, dx, dy):
        nonlocal current_x, current_y, current_dx, current_dy
        current_x, current_y, current_dx, current_dy = x, y, dx, dy

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
    assert np.allclose(current_x, 0)

    mouse_drag.handle_motion(mockevent(ax, 2, 2, button, 0))
    assert np.allclose(current_x, 2)
    assert np.allclose(current_y, 2)
    assert np.allclose(current_dx, 1)
    assert np.allclose(current_dy, 1)

    mouse_drag.handle_motion(mockevent(ax, 5, 5, button, 0))
    assert np.allclose(current_x, 5)
    assert np.allclose(current_y, 5)
    assert np.allclose(current_dx, 3)
    assert np.allclose(current_dy, 3)

    mouse_drag.button_release(mockevent(ax, 1, 1, button, 0))
    mouse_drag.handle_motion(mockevent(ax, 15, 15, button, 0))
    assert np.allclose(current_x, 5)
    assert np.allclose(current_y, 5)
    assert np.allclose(current_dx, 3)
    assert np.allclose(current_dy, 3)


def test_mouse_drag_lim_change(mockevent):
    """When limits change, the coordinate system changes. This test tests whether that is taken into account
    correctly."""
    plt.plot([1, 2, 3, 4, 5], [5, 6, 7, 8, 9])
    plt.xlim([0, 5])
    plt.ylim([5, 10])
    ax = plt.gca()

    current_x, current_y, current_dx, current_dy = 0, 0, 0, 0

    def callback(x, y, dx, dy):
        nonlocal current_x, current_y, current_dx, current_dy
        xl = plt.gca().get_xlim()
        plt.xlim([xl[0]+dx, xl[1] + dx])
        current_x, current_y, current_dx, current_dy = x, y, dx, dy

    mouse_drag = MouseDragCallback(ax, 1, callback)
    mouse_drag.button_down(mockevent(ax, 1, 1, 1, 0))
    mouse_drag.handle_motion(mockevent(ax, 1, 1, 1, 0))

    mouse_drag.handle_motion(mockevent(ax, 2, 2, 1, 0))
    assert np.allclose(current_x, 2)
    assert np.allclose(current_y, 2)
    assert np.allclose(current_dx, 1)
    assert np.allclose(current_dy, 1)

    mouse_drag.handle_motion(mockevent(ax, 2, 2, 1, 0))
    assert np.allclose(current_x, 2)
    assert np.allclose(current_y, 2)
    assert np.allclose(current_dx, -1)
    assert np.allclose(current_dy, 0)
