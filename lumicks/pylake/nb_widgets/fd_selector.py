import matplotlib.pyplot as plt
import numpy as np
import time


class FdRangeSelectorWidget:
    def __init__(self, fd_curve, axes=None, show=True):
        self._axes = axes if axes else plt.axes(label=f"lk_widget_{time.time()}")
        self.fd_curve = fd_curve
        self.time = self.fd_curve.f.timestamps
        self.connection_id = None
        self.current_range = []
        self.ranges = []
        self.open = show

        if show:
            self._axes.get_figure().canvas.mpl_connect('close_event', self.close)
            self.update_plot()

    def close(self, event):
        self.open = False

    def wait_for_input(self):
        while self.open:
            plt.pause(.0001)

    def to_seconds(self, timestamp):
        return (timestamp - self.time[0]) / 1e9

    def _add_point(self, fd_timestamp):
        """Adds a point intended to be a range endpoint. Once two points are selected the new range selection is
        committed and the range appended."""
        self.current_range.append(fd_timestamp)

        # We have a range selected. Append it to the list.
        if len(self.current_range) == 2:
            self.ranges.append(np.sort(self.current_range))
            self.current_range = []
            self.update_plot()

        # Draw a vertical line for some immediate visual feedback
        self._axes.axvline(self.to_seconds(fd_timestamp))
        self._axes.get_figure().canvas.draw()

    def _remove_range(self, fd_timestamp):
        """Removes a range if the provided timestamp falls inside it."""
        in_range = [start < fd_timestamp < stop for start, stop in self.ranges]
        selected = np.nonzero(in_range)[0]
        if len(selected) > 0:
            selected_ranges = [self.ranges[x] for x in selected]
            smallest_segment = np.argmin([stop - start for start, stop in selected_ranges])
            self.ranges.pop(selected[smallest_segment])
        self.update_plot()

    def handle_button_event(self, event):
        if event.inaxes != self._axes:
            return

        # Check if we aren't using a figure widget function like zoom.
        if event.canvas.widgetlock.locked():
            return

        event_timestamp = event.xdata * 1e9 + self.time[0]
        if event.button == 1:
            # Find the data point nearest to the event and determine its timestamp
            fd_timestamp = self.time[np.argmin(np.abs(self.time - event_timestamp))]
            if fd_timestamp:
                self._add_point(fd_timestamp)
        elif event.button == 3:
            # Find whether we clicked an existing range
            self._remove_range(event_timestamp)

    def connect_click_callback(self):
        """Connects a callback for clicking the axes"""
        self.connection_id = self._axes.get_figure().canvas.mpl_connect('button_press_event', self.handle_button_event)

    def disconnect_click_callback(self):
        """Disconnects the clicking callback if it exists"""
        if self.connection_id:
            self._axes.get_figure().canvas.mpl_disconnect(self.connection_id)

    def update_plot(self):
        self._axes.clear()

        for i, (t_start, t_end) in enumerate(self.ranges):
            t_start, t_end = self.to_seconds(t_start), self.to_seconds(t_end)
            self._axes.axvline(t_start)
            self._axes.axvline(t_end)
            self._axes.axvspan(t_start, t_end, alpha=0.15, color='blue')
            self._axes.text((t_start + t_end) / 2, 0, i)

        old_axis = plt.gca()
        plt.sca(self._axes)
        self.fd_curve.f.plot(linestyle='', marker='.', markersize=1)
        self.connect_click_callback()

        plt.ylabel('Force [pN]')
        plt.xlabel('Time [s]')
        plt.sca(old_axis)

    @property
    def fdcurves(self):
        return [self.fd_curve[start:stop] for start, stop in self.ranges]


class FdRangeSelector:
    def __init__(self, fd_curves):
        """Open widget to select regions in multiple F,d curves

        Parameters
        ----------
        fd_curves : dict
            Dictionary of `pylake.FdCurve`
        """

        if len(fd_curves) == 0:
            raise ValueError("F,d selector widget cannot open without a non-empty dictionary containing F,d curves.")

        try:
            import ipywidgets
        except ImportError:
            raise RuntimeError("This widget requires ipywidgets to be installed to work. Please install it.")

        if "ipympl" not in plt.get_backend():
            raise RuntimeError(("Please enable the widgets backend for this plot to work. You can do this by invoking "
                                "%matplotlib widget. Please note that you may have to restart the notebook kernel for "
                                "this to work."))

        plt.figure()
        self.axes = plt.axes()
        self.active_plot = None
        self.selectors = {key: FdRangeSelectorWidget(curve, self.axes, show=False) for key, curve in fd_curves.items()}
        keys = [key for key, curve in fd_curves.items()]
        ipywidgets.interact(self.update_plot, curve=ipywidgets.Dropdown(options=keys))

        self.update_plot(keys[0])

    def update_plot(self, curve):
        if curve:
            if self.active_plot:
                self.active_plot.disconnect_click_callback()

            self.active_plot = self.selectors[curve]
            self.active_plot.update_plot()

    @property
    def ranges(self):
        return {key: selector.ranges for key, selector in self.selectors.items()}

    @property
    def fdcurves(self):
        return {key: selector.fdcurves for key, selector in self.selectors.items()}

