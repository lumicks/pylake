import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from lumicks.pylake import track_greedy, filter_lines, refine_lines_centroid
from lumicks.pylake.nb_widgets.detail.mouse import MouseDragCallback
from lumicks.pylake.kymotracker.kymoline import KymoLineGroup, import_kymolinegroup_from_csv


class KymoWidget:
    def __init__(self, data, axis_aspect_ratio, min_length, use_widgets, output_filename, algorithm,
                 algorithm_parameters, **kwargs):
        """Create a widget for performing kymotracking.

        Parameters
        ----------
        data : array_like
            Image data.
        axis_aspect_ratio : float
            Desired aspect ratio of the viewport. Sometimes kymographs can be very long and thin. This helps you
            visualize them anyway.
        min_length : int
            Minimum length of a trace. Traces shorter than this are discarded.
        use_widgets : bool
            Add interactive widgets for interacting with algorithm parameters.
        output_filename : str
            Filename to save to and load from.
        algorithm : callable
            Kymotracking algorithm used
        algorithm_parameters : dict
            Parameters for the kymotracking algorithm.
        **kwargs
            Extra arguments forwarded to imshow.
        """
        # Forcing the aspect ratio only makes sense when the time axis is longer.
        self.axis_aspect_ratio = min(axis_aspect_ratio, data.shape[1] / data.shape[0]) if axis_aspect_ratio else None
        self.lines = KymoLineGroup([])
        self.plotted_lines = []
        self.min_length = min_length
        self.data = data
        self._label = None
        self._fig = None
        self._axes = None
        self.adding = True
        self.show_lines = True
        self.output_filename = output_filename

        self._dx = 0
        self._last_update = 0
        self.area_selector = None

        self._algorithm = algorithm
        self.algorithm_parameters = algorithm_parameters

        self.show(use_widgets=use_widgets, **kwargs)

    def track_kymo(self, click, release):
        """Handle mouse release event.

        Removes lines in a region, and traces new ones."""
        p1 = [int(click.xdata), int(click.ydata)]
        p2 = [int(release.xdata), int(release.ydata)]
        self.lines.remove_lines_in_rect([p1, p2])

        if self.adding:
            new_lines = self._track(rect=[p1, p2])
            self.lines.extend(new_lines)

        self.update_lines()

    def track_all(self):
        """Track all lines on the kymograph"""
        self.lines = self._track()
        self.update_lines()

    def _track(self, rect=None):
        return filter_lines(self._algorithm(self.data, **self.algorithm_parameters, rect=rect), self.min_length)

    def _connect_drag_callback(self):
        def set_xlim(_x, _y, dx, _dy):
            old_xlims = np.array(self._axes.get_xlim())
            self._dx = self._dx + dx

            # We don't need to update more than 30 times per second (1/30 = 0.033).
            if abs(time.time() - self._last_update) > 0.033:
                self._axes.set_xlim(old_xlims - self._dx)
                self._dx = 0
                self._last_update = time.time()

        MouseDragCallback(self._axes, 1, set_xlim)

    def update_lines(self):
        for line in self.plotted_lines:
            line.remove()
        self.plotted_lines = []

        if self.show_lines:
            self.plotted_lines = [self._axes.plot(line.time_idx, line.coordinate_idx, color='black', linewidth=5)[0]
                                  for line in self.lines]
            self.plotted_lines.extend(
                [self._axes.plot(line.time_idx, line.coordinate_idx, markersize=8)[0] for line in self.lines])

        self._fig.canvas.draw()

    def _save_from_ui(self):
        try:
            self.lines.save(self.output_filename,
                            sampling_width=int(np.ceil(.5 * self.algorithm_parameters["line_width"])))
            self._set_label(f"Saved {self.output_filename}")
        except (RuntimeError, IOError) as exception:
            self._set_label(str(exception))

    def save_lines(self, filename, dt=None, dx=None, delimiter=';', sampling_width=None):
        """Export KymoLineGroup to a csv file.

        Parameters
        ----------
        filename : str
            Filename to output kymograph traces to.
        dt : float
            Calibration for the time axis.
        dx : float
            Calibration for the coordinate axis.
        delimiter : str
            Which delimiter to use in the csv file.
        sampling_width : int or None
            When supplied, this will sample the source image around the kymograph line and export
            the summed intensity with the image. The value indicates the number of pixels in either direction
            to sum over.
        """
        self.lines.save(filename, dt, dx, delimiter, sampling_width)

    def _load_from_ui(self):
        try:
            self.lines = import_kymolinegroup_from_csv(self.output_filename, self.data)
            self.update_lines()
            self._set_label(f"Loaded {self.output_filename}")
        except (RuntimeError, IOError) as exception:
            self._set_label(str(exception))

    def _add_slider(self, description, name, tooltip, minimum, maximum, step_size=None, slider_type=None):
        import ipywidgets

        def set_value(value):
            self.algorithm_parameters[name] = value

        return ipywidgets.interactive(set_value,
                                      value=slider_type(
                                          description=description,
                                          description_tooltip=tooltip,
                                          min=minimum,
                                          max=maximum,
                                          step=step_size,
                                          value=self.algorithm_parameters[name]))

    def refine(self):
        self.lines = refine_lines_centroid(self.lines, self.algorithm_parameters["line_width"])
        self.update_lines()

    def create_algorithm_sliders(self):
        raise NotImplementedError("You should be using a class derived from this class to interact with the "
                                  "kymotracker algorithm")

    def _set_label(self, label):
        if self._label:
            self._label.value = label

    def _create_widgets(self):
        """Create widgets for setting kymotracking settings"""
        from IPython.display import display
        import ipywidgets

        if not max([backend in plt.get_backend() for backend in ("nbAgg", "ipympl")]):
            raise RuntimeError(("Please enable an interactive matplotlib backend for this widget to work. In jupyter "
                                "notebook you can do this by invoking either %matplotlib notebook or %matplotlib "
                                "widget (the latter requires ipympl to be installed). In Jupyter Lab only the latter "
                                "works. Please note that you may have to restart the notebook kernel for this to "
                                "work."))

        algorithm_sliders = self.create_algorithm_sliders()

        add_toggle = ipywidgets.interactive(
            lambda adding: setattr(self, "adding", adding),
            adding=ipywidgets.ToggleButton(
                description="Add lines",
                value=self.adding,
                disabled=False,
                button_style="",
                tooltip="Draw lines\n\nDisabling this will remove lines instead",
            ),
        )

        refine_button = ipywidgets.Button(description="Refine lines")
        refine_button.on_click(lambda button: self.refine())

        def set_show_lines(show_lines):
            self.show_lines = show_lines
            if self._fig:
                self.update_lines()

        show_lines_toggle = ipywidgets.interactive(
            set_show_lines,
            show_lines=ipywidgets.ToggleButton(
                description="Show Lines",
                value=self.show_lines,
                disabled=False,
                button_style="",
                tooltip="Show Lines\n\nDisabling this will hide all lines",
            ),
        )

        all_button = ipywidgets.Button(description="Track all", tooltip="Reset all lines and track all lines")
        all_button.on_click(lambda button: self.track_all())

        minimum_length = ipywidgets.interactive(
            lambda min_length: setattr(self, "min_length", min_length),
            min_length=ipywidgets.IntSlider(
                description="Min length",
                value=self.min_length,
                disabled=False,
                min=1,
                max=10,
                tooltip="Lines below this length are filtered out",
            ),
        )

        load_button = ipywidgets.Button(description="Load")
        load_button.on_click(lambda button: self._load_from_ui())

        save_button = ipywidgets.Button(description="Save")
        save_button.on_click(lambda button: self._save_from_ui())

        def set_fn(value):
            self.output_filename = value.new

        fn_widget = ipywidgets.Text(value=self.output_filename, description='File')
        fn_widget.observe(set_fn, 'value')

        self._label = ipywidgets.Label(value="")

        output = ipywidgets.Output()
        ui = ipywidgets.HBox(
            [
                ipywidgets.VBox([output]),
                ipywidgets.VBox(
                    [
                        ipywidgets.HBox([add_toggle, all_button]),
                        algorithm_sliders,
                        minimum_length,
                        ipywidgets.HBox([refine_button, show_lines_toggle]),
                        fn_widget,
                        ipywidgets.HBox([load_button, save_button]),
                        self._label
                    ], layout=ipywidgets.Layout(width='32%')
                ),
            ]
        )

        display(ui)

        with output:
            self._fig = plt.figure()
            self._axes = self._fig.add_subplot(111)

    def show(self, use_widgets, **kwargs):
        if self._fig:
            plt.close(self._fig)
            self.plotted_lines = []

        if use_widgets:
            self._create_widgets()
        else:
            self._fig = plt.figure()
            self._axes = self._fig.add_subplot(111)

        self._dx = 0
        self._last_update = time.time()
        plt.imshow(self.data, interpolation="nearest", **kwargs)

        if self.axis_aspect_ratio:
            self._axes.set_xlim([0, self.axis_aspect_ratio * self.data.shape[0]])

        # Prevents the axes from resetting every time new lines are drawn
        self._axes.autoscale(enable=False)
        plt.tight_layout()

        self.area_selector = RectangleSelector(self._axes, self.track_kymo,
                                               drawtype='box', useblit=True,
                                               button=[3],
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=False)

        self.update_lines()
        self._connect_drag_callback()


class KymoWidgetGreedy(KymoWidget):
    def __init__(self, data, axis_aspect_ratio=None, line_width=4, pixel_threshold=None, window=4, sigma=None, vel=0.0,
                 diffusion=0.0, sigma_cutoff=2.0, min_length=3, use_widgets=True, output_filename="kymotracks.txt",
                 **kwargs):
        """Create a widget for performing kymotracking.

        Parameters
        ----------
        data : array_like
            Image data.
        axis_aspect_ratio : float
            Desired aspect ratio of the viewport. Sometimes kymographs can be very long and thin. This helps you
            visualize them anyway.
        line_width : float
            Expected width of the particles in pixels.
        pixel_threshold : float
            Intensity threshold for the pixels. Local maxima above this intensity level will be designated as a line
            origin.
        window : int
            Number of kymograph lines in which the particle is allowed to disappear (and still be part of the same
            line).
        sigma : float or None
            Uncertainty in the particle position. This parameter will determine whether a peak in the next frame will be
            linked to this one. Increasing this value will make the algorithm tend to allow more positional variation in
            the lines. If none, the algorithm will use half the line width.
        vel : float
            Expected velocity of the traces in the image. This can be used for non-static particles that are expected to
            move at an expected rate (default: 0.0).
        diffusion : float
            Expected diffusion constant (default: 0.0). This parameter will influence whether a peak in the next frame
            will be connected to this one. Increasing this value will make the algorithm allow more positional variation
            in.
        sigma_cutoff : float
            Sets at how many standard deviations from the expected trajectory a particle no longer belongs to this
            trace.
            Lower values result in traces being more stringent in terms of continuing (default: 2.0).
        min_length : int
            Minimum length of a trace. Traces shorter than this are discarded.
        use_widgets : bool
            Add interactive widgets for interacting with algorithm parameters.
        output_filename : str
            Filename to save to and load from.
        """
        algorithm = track_greedy
        algorithm_parameters = {"line_width": line_width,
                                "pixel_threshold": np.percentile(data.flatten(), 98) if pixel_threshold is None
                                else pixel_threshold,
                                "window": window,
                                "sigma": .5 * line_width if sigma is None else sigma,
                                "vel": vel,
                                "diffusion": diffusion,
                                "sigma_cutoff": sigma_cutoff}

        super().__init__(data, axis_aspect_ratio, min_length, use_widgets, output_filename, algorithm,
                         algorithm_parameters, **kwargs)

    def create_algorithm_sliders(self):
        import ipywidgets

        window_slider = self._add_slider("Window", "window", "How many frames can a line disappear.", minimum=1,
                                         maximum=15, slider_type=ipywidgets.IntSlider)
        thresh_slider = self._add_slider("Threshold", "pixel_threshold", "Set the pixel threshold.", minimum=1,
                                         maximum=np.max(self.data), step_size=1, slider_type=ipywidgets.IntSlider)
        line_width_slider = self._add_slider(
            "Line width",
            "line_width",
            "Estimated spot width.",
            minimum=0.0,
            maximum=15.0,
            step_size=0.001,
            slider_type=ipywidgets.FloatSlider,
        )
        sigma_slider = self._add_slider(
            "Sigma",
            "sigma",
            "How much does the line fluctuate?",
            minimum=1.0,
            maximum=5.0,
            step_size=0.001,
            slider_type=ipywidgets.FloatSlider,
        )
        return ipywidgets.VBox([thresh_slider, line_width_slider, window_slider, sigma_slider])
