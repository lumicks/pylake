import pytest


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
