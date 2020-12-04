import pytest
import numpy as np


class MockEvent:
    class Canvas:
        class WidgetLock:
            def __init__(self, locked):
                self.am_locked = locked

            def locked(self):
                return self.am_locked

        def __init__(self, locked):
            self.widgetlock = self.WidgetLock(locked)

    def __init__(self, axis, x, y, button, widget_lock):
        self.inaxes = axis
        self.xdata = x
        self.ydata = y
        self.button = button
        self.canvas = self.Canvas(widget_lock)


@pytest.fixture
def mockevent():
    return lambda axis, x, y, button, widget_lock: MockEvent(axis, x, y, button, widget_lock)


@pytest.fixture
def region_select():
    def region(xs, ys, xe, ye):
        return MockEvent(0, ys, xs, 0, 0), MockEvent(0, ye, xe, 0, 0)

    return lambda xs, ys, xe, ye: region(xs, ys, xe, ye)


@pytest.fixture(scope="session")
def kymograph():
    data = np.ones((20, 30))
    data[5, 5:15] = 6
    data[8, 10:25] = 8
    data[12, 5:7] = 10
    data[12, 9:20] = 10
    return data
