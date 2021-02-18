import matplotlib.pyplot as plt
import numpy as np
import time


class BaseRangeSelectorWidget:
    """Base class for range selection widgets.
    Takes care of setting up connections and internal event handling.

    Parameters
    __________
    axes : matplotlib.axes.Axes
        Axes instance used for plotting
    show : bool
        Controls if plot is rendered immediately
    range_conversion_function : function
        Function to convert from units of the underlying data to plotting units,
        for example: timestamps -> seconds in the case of a Slice
        default simply returns the value unchanged.
    """

    def __init__(self, axes=None, show=True, range_conversion_fcn=lambda point: point):
        self._axes = axes if axes else plt.axes(label=f"lk_slice_widget_{time.time()}")
        self.connection_id = None
        self.current_range = []
        self._ranges = []
        self.range_conversion_fcn = range_conversion_fcn
        self.open = show

        if show:
            self._axes.get_figure().canvas.mpl_connect("close_event", self.close)
            self.update_plot()

    @property
    def _xdata(self):
        raise NotImplementedError

    @property
    def ranges(self):
        """Returns selected list of timestamp ranges"""
        return np.copy(self._ranges)

    def close(self, event):
        self.open = False

    def wait_for_input(self):
        while self.open:
            plt.pause(0.0001)

    def _add_point(self, point):
        """Adds a point intended to be a range endpoint. Once two points are selected the new range selection is
        committed and the range appended."""
        self.current_range.append(point)

        # We have a range selected. Append it to the list.
        if len(self.current_range) == 2:
            self._ranges.append(np.sort(self.current_range))
            self.current_range = []
            self.update_plot()

        # Draw a vertical line for some immediate visual feedback
        self._axes.axvline(self.range_conversion_fcn(point))
        self._axes.get_figure().canvas.draw()

    def _remove_range(self, point):
        """Removes a range if the provided timestamp falls inside it."""
        in_range = [start < point < stop for start, stop in self._ranges]
        selected = np.nonzero(in_range)[0]
        if len(selected) > 0:
            selected_ranges = [self._ranges[x] for x in selected]
            smallest_segment = np.argmin([stop - start for start, stop in selected_ranges])
            self._ranges.pop(selected[smallest_segment])
        self.update_plot()

    def handle_button_event(self, event):
        if event.inaxes != self._axes:
            return

        # Check if we aren't using a figure widget function like zoom.
        if event.canvas.widgetlock.locked():
            return

        if event.button == 1:
            # Find the data point nearest to the event and determine its timestamp
            point = self._xdata[np.argmin(np.abs(self._xdata - event.xdata))]
            if point:
                self._add_point(point)
        elif event.button == 3:
            # Find whether we clicked an existing range
            self._remove_range(event.xdata)

    def connect_click_callback(self):
        """Connects a callback for clicking the axes"""
        self.connection_id = self._axes.get_figure().canvas.mpl_connect(
            "button_press_event", self.handle_button_event
        )

    def disconnect_click_callback(self):
        """Disconnects the clicking callback if it exists"""
        if self.connection_id:
            self._axes.get_figure().canvas.mpl_disconnect(self.connection_id)

    def update_plot(self):
        self._axes.clear()

        for i, (t_start, t_end) in enumerate(self._ranges):
            t_start, t_end = self.range_conversion_fcn(t_start), self.range_conversion_fcn(t_end)
            self._axes.axvline(t_start)
            self._axes.axvline(t_end)
            self._axes.axvspan(t_start, t_end, alpha=0.15, color="blue")
            self._axes.text((t_start + t_end) / 2, 0, i)

        old_axis = plt.gca()
        plt.sca(self._axes)
        self._plot_data()
        self.connect_click_callback()
        plt.sca(old_axis)

    def _plot_data(self):
        raise NotImplementedError


class SliceRangeSelectorWidget(BaseRangeSelectorWidget):
    def __init__(self, channel_slice, axes=None, show=True):
        self.slice = channel_slice
        self.time = self.slice.timestamps
        super().__init__(axes, show, self.to_seconds)

    @property
    def _xdata(self):
        return self.time

    def to_seconds(self, timestamp):
        return (timestamp - self.time[0]) / 1e9

    def handle_button_event(self, event):
        event.xdata = event.xdata * 1e9 + self.time[0]  # convert seconds to timestamp
        super().handle_button_event(event)

    def _plot_data(self):
        self.slice.plot(linestyle="", marker=".", markersize=1)

    @property
    def slices(self):
        """Return list of selected slices of data as `lumicks.pylake.Slice`"""
        return [self.slice[start:stop] for start, stop in self._ranges]


class FdTimeRangeSelectorWidget(SliceRangeSelectorWidget):
    def __init__(self, fd_curve, axes=None, show=True):
        self.fd_curve = fd_curve
        super().__init__(fd_curve.f, axes, show)

    @property
    def fdcurves(self):
        """Return list of selected fdcurves of data as `lumicks.pylake.FdCurve`"""
        return [self.fd_curve[start:stop] for start, stop in self._ranges]


class FdDistanceRangeSelectorWidget(BaseRangeSelectorWidget):
    def __init__(self, fd_curve, axes=None, show=True, max_gap=0):
        self.fd_curve = fd_curve
        self._max_gap = max_gap
        super().__init__(axes, show)

    @property
    def _xdata(self):
        return self.fd_curve.d.data

    def _plot_data(self):
        self.fd_curve.plot_scatter(s=2)

    @property
    def fdcurves(self):
        """Return list of selected fdcurves of data as `lumicks.pylake.FdCurve`"""
        return [
            self.fd_curve._sliced_by_distance(min_dist, max_dist, self._max_gap)
            for min_dist, max_dist in self._ranges
        ]


class BaseRangeSelector:
    def __init__(self, fd_curves):
        """Open widget to select regions in multiple F,d curves

        Parameters
        ----------
        fd_curves : dict
            Dictionary of `lumicks.pylake.FdCurve`
        """
        import ipywidgets

        if len(fd_curves) == 0:
            raise ValueError(
                "F,d selector widget cannot open without a non-empty dictionary containing F,d curves."
            )

        if not any(backend in plt.get_backend() for backend in ("nbAgg", "ipympl")):
            raise RuntimeError(
                (
                    "Please enable an interactive matplotlib backend for this plot to work. In jupyter"
                    "notebook you can do this by invoking either %matplotlib notebook or %matplotlib "
                    "widget (the latter requires ipympl to be installed). In Jupyter Lab only the latter "
                    "works. Please note that you may have to restart the notebook kernel for this to "
                    "work."
                )
            )

        plt.figure()
        self.axes = plt.axes()
        self.active_plot = None
        self.selectors = {key: self._add_widget(curve) for key, curve in fd_curves.items()}
        keys = [key for key, curve in fd_curves.items()]
        ipywidgets.interact(self.update_plot, curve=ipywidgets.Dropdown(options=keys))

        self.update_plot(keys[0])

    def _add_widget(self, curve):
        raise NotImplementedError

    def update_plot(self, curve):
        if curve:
            if self.active_plot:
                self.active_plot.disconnect_click_callback()

            self.active_plot = self.selectors[curve]
            self.active_plot.update_plot()

    @property
    def ranges(self):
        """Return list of range timestamps"""
        return {key: selector.ranges for key, selector in self.selectors.items()}

    @property
    def fdcurves(self):
        """Return list of selected fdcurves of data as `lumicks.pylake.FdCurve`"""
        return {key: selector.fdcurves for key, selector in self.selectors.items()}


class FdRangeSelector(BaseRangeSelector):
    def _add_widget(self, curve):
        return FdTimeRangeSelectorWidget(curve, self.axes, show=False)


class FdDistanceRangeSelector(BaseRangeSelector):
    def __init__(self, fd_curves, max_gap=3):
        self._max_gap = max_gap
        super().__init__(fd_curves)

    def _add_widget(self, curve):
        return FdDistanceRangeSelectorWidget(curve, self.axes, show=False, max_gap=self._max_gap)
