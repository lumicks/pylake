from dataclasses import dataclass


@dataclass
class MouseDragEvent:
    x: float
    y: float
    dx: float
    dy: float


class MouseDragCallback:
    def __init__(self, axes, button, drag_callback, press_callback=None, release_callback=None):
        """Callback handler to handle mouse dragging for a particular button. Callback is called
        with x, y, dx, dy as arguments.

        Parameters
        ----------
        axes : :class:`~matplotlib.axes.Axes`
            Axes to attach the callback to.
        button : int
            Which button to respond to (0: Left, 3: Right).
        drag_callback : callable
            Which function to call when dragging takes place. Note that this function should have
            the following input signature: fun(MouseDragEvent). MouseDragEvent contains x, y,
            dx and dy which are all defined in data coordinates.
        press_callback : callable
            Function to call when drag is initiated. Must return True if it is a valid drag.
        release_callback : callable
            Function to call when drag is released.
        """
        self._axes = axes
        self._button = button
        self._drag_callback = drag_callback
        self._release_callback = release_callback
        self._press_callback = press_callback
        self.dragging = False
        self.x_last = 0
        self.y_last = 0

        self._callback_ids = {"press": None, "release": None, "motion": None}
        self.set_active(True)

    def set_active(self, state):
        # Disconnect existing callbacks if present
        for callback_name in self._callback_ids:
            if self._callback_ids[callback_name] is not None:
                self._axes.get_figure().canvas.mpl_disconnect(self._callback_ids[callback_name])
                self._callback_ids[callback_name] = None

        if state:
            self._callback_ids["press"] = self._axes.get_figure().canvas.mpl_connect(
                "button_press_event", lambda event: self.button_down(event)
            )
            self._callback_ids["release"] = self._axes.get_figure().canvas.mpl_connect(
                "button_release_event", lambda event: self.button_release(event)
            )
            self._callback_ids["motion"] = self._axes.get_figure().canvas.mpl_connect(
                "motion_notify_event", lambda event: self.handle_motion(event)
            )
        else:
            self.dragging = False

    def button_down(self, event):
        if event.inaxes == self._axes and not event.canvas.widgetlock.locked():
            if event.button == self._button:
                # If there's a starting callback, then it decides whether this will initiate a drag.
                self.dragging = (
                    self._press_callback(MouseDragEvent(event.xdata, event.ydata, 0, 0))
                    if self._press_callback
                    else True
                )
                self.x_last = event.xdata
                self.y_last = event.ydata

    def button_release(self, event):
        if event.button == self._button:
            if self.dragging:
                self.dragging = False
                if self._release_callback:
                    self._release_callback(
                        MouseDragEvent(
                            event.xdata,
                            event.ydata,
                            event.xdata - self.x_last,
                            event.ydata - self.y_last,
                        )
                    )

    def handle_motion(self, event):
        # There's an issue that if you drag outside the figure, the release is not detected. Hence
        # the event.inaxes != self._axes also leads to dragging being disabled
        if event.button != self._button or event.inaxes != self._axes:
            self.dragging = False

        if self.dragging:
            dx = event.xdata - self.x_last
            dy = event.ydata - self.y_last

            # Convert to mouse position
            to_display_position = self._axes.transData
            old_position = to_display_position.transform((event.xdata, event.ydata))

            self._drag_callback(MouseDragEvent(event.xdata, event.ydata, dx, dy))

            # User may have changed the limits, so convert back to data position for the dx and dy
            # in data space
            to_data_position = self._axes.transData.inverted()
            x_new, y_new = to_data_position.transform(old_position)

            self.x_last = x_new
            self.y_last = y_new
