from lumicks.pylake import FdRangeSelectorWidget, FdRangeSelector
from lumicks.pylake.fdcurve import FDCurve
from lumicks.pylake.channel import TimeSeries, Slice
from matplotlib.testing.decorators import cleanup
import pytest
import numpy as np


def make_mock_fd(force, distance, start=0, dt=600e9):
    """Mock FD curve which is not attached to an actual file, timestamps start at `start`"""
    assert len(force) == len(distance)
    fd = FDCurve(file=None, start=None, stop=None, name="")
    timestamps = int(dt) * np.arange(len(force)) + start
    fd._force_cache = Slice(TimeSeries(force, timestamps))
    fd._distance_cache = Slice(TimeSeries(distance, timestamps))
    return fd


class MockEvent:
    class Canvas:
        class WidgetLock:
            def __init__(self, locked):
                self.am_locked = locked

            def locked(self):
                return self.am_locked

        def __init__(self, locked):
            self.widgetlock = self.WidgetLock(locked)

    def __init__(self, axis, x, button, widget_lock):
        self.inaxes = axis
        self.xdata = x
        self.button = button
        self.canvas = self.Canvas(widget_lock)


@cleanup
def test_selector_widget():
    start_point = int(2500e9)
    dt = int(600e9)
    fd_curve = make_mock_fd([0, 1, 2, 3], [0, 1, 2, 3], start=start_point, dt=dt)

    selector = FdRangeSelectorWidget(fd_curve, show=False)

    assert selector.current_range == []
    selector._add_point(750)
    assert selector.current_range == [750]
    selector._add_point(450)
    assert selector.current_range == []
    assert np.allclose(selector.ranges, [[450, 750]])
    selector._add_point(850)
    selector._add_point(950)
    assert selector.current_range == []
    assert np.allclose(selector.ranges, [[450, 750], [850, 950]])

    assert selector.to_seconds(2500e9) == 0
    selector.update_plot()

    selector = FdRangeSelectorWidget(fd_curve)

    # Add a point
    event = MockEvent(selector._ax, 650, 1, False)
    selector.handle_button_event(event)
    assert selector.current_range == [start_point + dt]

    # Widget is locked, do not add!
    event = MockEvent(selector._ax, 650, 1, True)
    selector.handle_button_event(event)
    assert selector.current_range == [start_point + dt]

    # Not the axis we own, do not add!
    event = MockEvent(5, 650, 1, False)
    selector.handle_button_event(event)
    assert selector.current_range == [start_point + dt]

    # Not the axis we own, do not add!
    event = MockEvent(selector._ax, 950, 1, False)
    selector.handle_button_event(event)
    assert selector.current_range == []
    assert np.allclose(selector.ranges, [start_point + dt, start_point + 2*dt])


@cleanup
def test_multi_selector_widget():
    fd_curve1 = make_mock_fd([0, 1, 2, 3], [0, 1, 2, 3], start=int(2500e9))
    fd_curve2 = make_mock_fd([2, 3, 4, 5], [2, 3, 4, 5], start=int(2500e9))

    with pytest.raises(RuntimeError):
        FdRangeSelector([fd_curve1, fd_curve2])
