from lumicks.pylake.fdcurve import FDCurve
import matplotlib.pyplot as plt
import numpy as np
import time


class FdRangeSelectorWidget:
    def __init__(self, fd_curve, ax=None, show=True):
        self._ax = ax if ax else plt.axes(label=f"lk_widget_{time.time()}")
        self.fd_curve = fd_curve
        self.time = self.fd_curve.f.timestamps
        self.connection_id = None
        self.current_range = []
        self.ranges = []
        self.open = show

        if show:
            self._ax.get_figure().canvas.mpl_connect('close_event', self.close)
            self.update_plot()

    def close(self, event):
        self.open = False

    def wait_for_input(self):
        while self.open:
            plt.pause(.0001)

    def to_seconds(self, timestamp):
        return (timestamp - self.time[0]) / 1e9

    def _add_point(self, fd_timestamp):
        self.current_range.append(fd_timestamp)

        # We have a range selected. Append it to the list.
        if len(self.current_range) == 2:
            self.ranges.append(np.sort(self.current_range))
            self.current_range = []
            self.update_plot()

        # Draw a vertical line for some immediate visual feedback
        plt.axvline(self.to_seconds(fd_timestamp))
        self._ax.get_figure().canvas.draw()

    def handle_button_event(self, event):
        if event.inaxes != self._ax:
            return

        # Check if we aren't using a figure widget function like zoom.
        if event.canvas.widgetlock.locked():
            return

        # Find the data point nearest to the event and determine its timestamp
        fd_timestamp = self.time[np.argmin(np.abs(self.time - (event.xdata * 1e9 + self.time[0])))]
        if fd_timestamp:
            self._add_point(fd_timestamp)

    def disconnect_click_callback(self):
        """Disconnects the clicking callback if it exists"""
        if self.connection_id:
            self._ax.get_figure().canvas.mpl_disconnect(self.connection_id)

    def connect_click_callback(self):
        """Connects a callback for clicking the axes"""
        self.connection_id = self._ax.get_figure().canvas.mpl_connect('button_press_event', self.handle_button_event)

    def update_plot(self):
        self._ax.clear()

        for i, (t_start, t_end) in enumerate(self.ranges):
            t_start, t_end = self.to_seconds(t_start), self.to_seconds(t_end)
            self._ax.axvline(t_start)
            self._ax.axvline(t_end)
            self._ax.axvspan(t_start, t_end, alpha=0.15, color='blue')
            plt.text((t_start + t_end) / 2, 0, i)

        self.fd_curve.f.plot(linestyle='', marker='.', markersize=1)
        self.connect_click_callback()

        plt.ylabel('Force [pN]')
        plt.xlabel('Time [s]')

    @property
    def fdcurves(self):
        return [self.fd_curve[start:stop] for start, stop in self.ranges]


class FdRangeSelector:
    def __init__(self, fd_curves):
        try:
            import ipywidgets
        except ImportError:
            raise RuntimeError("This widget requires ipywidgets to be installed to work. Please install it.")

        if "ipympl" not in plt.get_backend():
            raise RuntimeError(("Please enable the widgets backend for this plot to work. You can do this by invoking "
                                "%matplotlib widget. Please note that you may have to restart the notebook kernel for "
                                "this to work."))

        plt.figure()
        self.ax = plt.axes()
        self.active_plot = None
        self.selectors = {key: FdRangeSelectorWidget(curve, self.ax, show=False) for key, curve in fd_curves.items()}
        self.keys = [key for key, curve in fd_curves.items()]
        ipywidgets.interact(self.update_plot,
                            curve_idx=ipywidgets.IntSlider(min=0, max=len(fd_curves) - 1, step=1, value=0))

        self.update_plot(0)

    def update_plot(self, curve_idx):
        if self.active_plot:
            self.active_plot.disconnect_click_callback()

        self.active_plot = self.selectors[self.keys[curve_idx]]
        self.active_plot.update_plot()

    @property
    def ranges(self):
        return {key: selector.ranges for key, selector in self.selectors.items()}

    @property
    def fdcurves(self):
        return {key: selector.fdcurves for key, selector in self.selectors.items()}

