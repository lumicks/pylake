class MouseDragCallback:
    def __init__(self, axes, button, callback):
        """Callback handler to handle mouse dragging for a particular button. Callback is called with x, y, dx, dy as
        arguments.

        Parameters
        ----------
        axes : :class:`~matplotlib.axes.Axes`
            Axes to attach the callback to.
        button : int
            Which button to respond to (0: Left, 1: Middle, 2: Right).
        callback : callable
            Which function to call when dragging takes place. Note that this function should have the following input
            signature: fun(x, y, dx, dy) where x, y are the location in data space and dx and dy are the dragging
            distances in data space.
        """
        self._axes = axes
        self._button = button
        self._callback = callback
        self.dragging = 0
        self.x_last = 0
        self.y_last = 0
        self._axes.get_figure().canvas.mpl_connect("button_press_event", lambda event: self.button_down(event))
        self._axes.get_figure().canvas.mpl_connect("button_release_event", lambda event: self.button_release(event))
        self._axes.get_figure().canvas.mpl_connect("motion_notify_event", lambda event: self.handle_motion(event))

    def button_down(self, event):
        if event.inaxes == self._axes and not event.canvas.widgetlock.locked():
            if event.button == self._button:
                self.dragging = True
                self.x_last = event.xdata
                self.y_last = event.ydata

    def button_release(self, event):
        if event.button == self._button:
            self.dragging = False

    def handle_motion(self, event):
        # There's an issue that if you drag outside the figure, the release is not detected. Hence the
        # event.inaxes != self._axes also leads to dragging being disabled
        if event.button != self._button or event.inaxes != self._axes:
            self.dragging = False

        if self.dragging:
            dx = event.xdata - self.x_last
            dy = event.ydata - self.y_last

            # Convert to mouse position
            to_display_position = self._axes.transData
            old_position = to_display_position.transform((event.xdata, event.ydata))

            self._callback(event.xdata, event.ydata, dx, dy)

            # User may have changed the limits, so convert back to data position for the dx and dy in data space
            to_data_position = self._axes.transData.inverted()
            x_new, y_new = to_data_position.transform(old_position)

            self.x_last = x_new
            self.y_last = y_new
