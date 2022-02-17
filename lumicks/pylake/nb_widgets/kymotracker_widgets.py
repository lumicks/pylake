import time
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from matplotlib.widgets import RectangleSelector
from lumicks.pylake.kymotracker.detail.calibrated_images import CalibratedKymographChannel
from lumicks.pylake.kymotracker.kymotracker import track_greedy
from lumicks.pylake import filter_lines, refine_lines_centroid
from lumicks.pylake.nb_widgets.detail.mouse import MouseDragCallback
from lumicks.pylake.kymotracker.kymoline import KymoLineGroup, import_kymolinegroup_from_csv
from lumicks.pylake.nb_widgets.detail.undostack import UndoStack


class KymoWidget:
    def __init__(
        self,
        kymo,
        channel,
        axis_aspect_ratio,
        min_length,
        use_widgets,
        output_filename,
        algorithm,
        algorithm_parameters,
        min_length_range,
        **kwargs,
    ):
        """Create a widget for performing kymotracking.

        Parameters
        ----------
        kymo : lumicks.pylake.Kymo
            Kymograph.
        channel : str
            Kymograph channel to use.
        axis_aspect_ratio : float
            Desired aspect ratio of the viewport. Sometimes kymographs can be very long and thin.
            This helps you visualize them.
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
        minimum_length_range : tuple of int
            Range of the minimum length parameter. Should be of the form (lower bound, upper bound).
        **kwargs
            Extra arguments forwarded to imshow.
        """
        calibrated_image = CalibratedKymographChannel.from_kymo(kymo, channel)

        # Forcing the aspect ratio only makes sense when the time axis is longer.
        self.axis_aspect_ratio = (
            min(
                axis_aspect_ratio,
                calibrated_image.to_position(calibrated_image.data.shape[1])
                / calibrated_image.to_seconds(calibrated_image.data.shape[0]),
            )
            if axis_aspect_ratio
            else None
        )
        self._lines_history = UndoStack(KymoLineGroup([]))
        self.plotted_lines = []
        self.min_length = min_length
        self._min_length_range = min_length_range
        self._kymo = kymo
        self._channel = channel
        self._label = None
        self._fig = None
        self._axes = None
        self.adding = True
        self.show_lines = True
        self.output_filename = output_filename

        self._dx = 0
        self._last_update = 0
        self._area_selector = None
        self._line_connector = None

        self._algorithm = algorithm
        self.algorithm_parameters = algorithm_parameters

        self.show(use_widgets=use_widgets, **kwargs)

    @property
    def lines(self):
        return self._lines_history.state

    @lines.setter
    def lines(self, new_lines):
        self._lines_history.state = new_lines

    @property
    def _line_width_pixels(self):
        calibrated_image = CalibratedKymographChannel.from_kymo(self._kymo, self._channel)
        return np.ceil(self.algorithm_parameters["line_width"] / calibrated_image._pixel_size)

    def track_kymo(self, click, release):
        """Handle mouse release event.

        Removes lines in a region, and traces new ones."""
        p1 = [click.xdata, click.ydata]
        p2 = [release.xdata, release.ydata]

        # Explicit copy to make modifications. Current state pushed to undo stack on assignment.
        lines = copy(self.lines)
        lines.remove_lines_in_rect([p1, p2])

        if self.adding:
            new_lines = self._track(rect=[p1, p2])
            lines.extend(new_lines)

        self.lines = lines
        self.update_lines()

    def track_all(self):
        """Track all lines on the kymograph"""
        self.lines = self._track()
        self.update_lines()

    def _track(self, rect=None):
        return filter_lines(
            self._algorithm(
                self._kymo,
                self._channel,
                **self.algorithm_parameters,
                rect=rect,
            ),
            self.min_length,
        )

    def _connect_drag_callback(self):
        def set_xlim(drag_event):
            # Callback for dragging the field of view
            old_xlims = np.array(self._axes.get_xlim())
            self._dx = self._dx + drag_event.dx

            # We don't need to update more than 30 times per second (1/30 = 0.033).
            if abs(time.time() - self._last_update) > 0.033:
                self._axes.set_xlim(old_xlims - self._dx)
                self._dx = 0
                self._last_update = time.time()

        MouseDragCallback(self._axes, 1, set_xlim)

    def _get_scale(self):
        """Get scaling of the image axes"""
        return tuple(lims[1] - lims[0] for lims in (self._axes.get_xlim(), self._axes.get_ylim()))

    def _connect_line_callback(self):
        cutoff_radius = 0.05  # We use a connection cutoff of 5% of the axis ranges
        clicked_line_info = None
        plotted_line = None
        nodes = None

        def get_node_info():
            nodes = [
                np.vstack(
                    (
                        np.full(len(track), j),  # track index
                        np.arange(len(track)),  # node index within track
                        track.seconds,  # x-coordinate
                        track.position,  # y-coordinate
                    )
                )
                for j, track in enumerate(self.lines)
            ]
            return np.hstack(nodes).T

        def get_nearest(x, y):
            nonlocal nodes
            ref_position = np.array([x, y])
            squared_dist = np.sum(((ref_position - nodes[:, -2:]) / self._get_scale()) ** 2, 1)
            idx = np.argmin(squared_dist)
            return np.sqrt(squared_dist[idx]), idx

        def initiate_line(event):
            nonlocal nodes, clicked_line_info
            if len(self.lines) == 0:
                return

            nodes = get_node_info()
            distance, idx = get_nearest(event.x, event.y)

            if distance < cutoff_radius:
                line_index, *line_info = nodes[idx]
                clicked_line_info = [self.lines[int(line_index)], *line_info]
                return True

        def drag_line(event):
            nonlocal clicked_line_info, plotted_line
            if plotted_line:
                plotted_line.remove()
                plotted_line = None

            line, node_index, *_ = clicked_line_info
            plotted_line, *_ = self._axes.plot(
                [line.seconds[int(node_index)], event.x],
                [line.position[int(node_index)], event.y],
                "r",
            )

        def finalize_line(event):
            nonlocal clicked_line_info, plotted_line
            if plotted_line:
                plotted_line.remove()
                plotted_line = None

            distance, idx = get_nearest(event.x, event.y)
            if distance < cutoff_radius:
                # Explicit copy to make modifications. Current state pushed to undo stack on
                # assignment.
                lines = copy(self.lines)

                line_index, *line_info = nodes[idx]
                released_line_info = [self.lines[int(line_index)], *line_info]

                clicked = [clicked_line_info, released_line_info]
                clicked.sort(key=lambda x: x[2])  # by time
                lines._merge_lines(*clicked[0][:2], *clicked[1][:2])

                self.lines = lines
                self.update_lines()

        self._line_connector = MouseDragCallback(
            self._axes, 3, drag_line, press_callback=initiate_line, release_callback=finalize_line
        )
        self._line_connector.set_active(False)

    def update_lines(self):
        for line in self.plotted_lines:
            line.remove()
        self.plotted_lines = []

        if self.show_lines:
            self.plotted_lines = [
                self._axes.plot(line.seconds, line.position, color="black", linewidth=5)[0]
                for line in self.lines
            ]
            self.plotted_lines.extend(
                [
                    self._axes.plot(line.seconds, line.position, markersize=8)[0]
                    for line in self.lines
                ]
            )

        self._fig.canvas.draw()

    def _save_from_ui(self):
        try:
            self.save_lines(
                self.output_filename,
                sampling_width=int(np.ceil(0.5 * self._line_width_pixels)),
            )
            self._set_label(f"Saved {self.output_filename}")
        except (RuntimeError, IOError) as exception:
            self._set_label(str(exception))

    def save_lines(self, filename, delimiter=";", sampling_width=None):
        """Export KymoLineGroup to a csv file.

        Parameters
        ----------
        filename : str
            Filename to output kymograph traces to.
        delimiter : str
            Which delimiter to use in the csv file.
        sampling_width : int or None
            When supplied, this will sample the source image around the kymograph line and export
            the summed intensity with the image. The value indicates the number of pixels in either direction
            to sum over.
        """
        self.lines.save(filename, delimiter, sampling_width)

    def _load_from_ui(self):
        try:
            self.lines = import_kymolinegroup_from_csv(
                self.output_filename, self._kymo, self._channel
            )
            self.update_lines()
            self._set_label(f"Loaded {self.output_filename}")
        except (RuntimeError, IOError) as exception:
            self._set_label(str(exception))

    def _add_slider(
        self, description, name, tooltip, minimum, maximum, step_size=None, slider_type=None
    ):
        import ipywidgets

        def set_value(value):
            self.algorithm_parameters[name] = value

        return ipywidgets.interactive(
            set_value,
            value=slider_type(
                description=description,
                description_tooltip=tooltip,
                min=minimum,
                max=maximum,
                step=step_size,
                value=self.algorithm_parameters[name],
            ),
        )

    def refine(self):
        if self.lines:
            self.lines = refine_lines_centroid(self.lines, self._line_width_pixels)
            self.update_lines()
        else:
            self._set_label("You need to track or load kymograph lines before you can refine them")

    def create_algorithm_sliders(self):
        raise NotImplementedError(
            "You should be using a class derived from this class to interact with the "
            "kymotracker algorithm"
        )

    def _set_label(self, label):
        if self._label:
            self._label.value = label

    def _create_widgets(self):
        """Create widgets for setting kymotracking settings"""
        from IPython.display import display
        import ipywidgets

        if not max([backend in plt.get_backend() for backend in ("nbAgg", "ipympl")]):
            raise RuntimeError(
                (
                    "Please enable an interactive matplotlib backend for this widget to work. In jupyter "
                    "notebook you can do this by invoking either %matplotlib notebook or %matplotlib "
                    "widget (the latter requires ipympl to be installed). In Jupyter Lab only the latter "
                    "works. Please note that you may have to restart the notebook kernel for this to "
                    "work."
                )
            )

        algorithm_sliders = self.create_algorithm_sliders()

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

        all_button = ipywidgets.Button(
            description="Track all", tooltip="Reset all lines and track all lines"
        )
        all_button.on_click(lambda button: self.track_all())

        def undo(button):
            self._lines_history.undo()
            self.update_lines()

        def redo(button):
            self._lines_history.redo()
            self.update_lines()

        undo_button = ipywidgets.Button(description="Undo", tooltip="Undo")
        undo_button.on_click(undo)
        redo_button = ipywidgets.Button(description="Redo", tooltip="Redo")
        redo_button.on_click(redo)

        minimum_length = ipywidgets.interactive(
            lambda min_length: setattr(self, "min_length", min_length),
            min_length=ipywidgets.IntSlider(
                description="Min length",
                value=self.min_length,
                disabled=False,
                min=self._min_length_range[0],
                max=self._min_length_range[1],
                tooltip="Minimum number of frames a spot has to be detected in to be considered",
            ),
        )

        load_button = ipywidgets.Button(description="Load")
        load_button.on_click(lambda button: self._load_from_ui())

        save_button = ipywidgets.Button(description="Save")
        save_button.on_click(lambda button: self._save_from_ui())

        def set_fn(value):
            self.output_filename = value.new

        fn_widget = ipywidgets.Text(value=self.output_filename, description="File")
        fn_widget.observe(set_fn, "value")

        self._mode = ipywidgets.RadioButtons(
            options=["Track lines", "Remove lines", "Connect lines"],
            disabled=False,
            tooltip="Choose between adding/removing tracked lines and connecting existing ones",
            layout=ipywidgets.Layout(flex_flow="row"),
        )
        self._mode.observe(self._select_state, "value")
        self._label = ipywidgets.Label(value="")

        output = ipywidgets.Output()
        ui = ipywidgets.HBox(
            [
                ipywidgets.VBox([output]),
                ipywidgets.VBox(
                    [
                        all_button,
                        algorithm_sliders,
                        minimum_length,
                        ipywidgets.HBox([refine_button, show_lines_toggle]),
                        fn_widget,
                        ipywidgets.HBox([undo_button, redo_button]),
                        ipywidgets.HBox([load_button, save_button]),
                        self._mode,
                        self._label,
                    ],
                    layout=ipywidgets.Layout(width="32%"),
                ),
            ]
        )

        display(ui)

        with output:
            self._fig = plt.figure()
            self._axes = self._fig.add_subplot(111)

    def _select_state(self, value):
        """Select a different state to operate the widget in. Note that the input argument is value
        because it is hooked up to a ToggleButton directly"""
        self._area_selector.set_active(False)
        self._line_connector.set_active(False)

        if value["new"] == "Track lines":
            self._set_label(f"Drag right mouse button to track lines")
            self._area_selector.set_active(True)
            self.adding = True
        elif value["new"] == "Remove lines":
            self._set_label(f"Drag right mouse button to remove lines")
            self._area_selector.set_active(True)
            self.adding = False
        elif value["new"] == "Connect lines":
            self._set_label(f"Drag right mouse button to connect lines")
            self._line_connector.set_active(True)

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
        calibrated_image = CalibratedKymographChannel.from_kymo(self._kymo, self._channel)
        calibrated_image.plot(interpolation="nearest", **kwargs)

        if self.axis_aspect_ratio:
            self._axes.set_xlim(
                [
                    0,
                    self.axis_aspect_ratio
                    * calibrated_image.to_seconds(calibrated_image.data.shape[0]),
                ]
            )

        # Prevents the axes from resetting every time new lines are drawn
        self._axes.autoscale(enable=False)
        plt.tight_layout()

        self._area_selector = RectangleSelector(
            self._axes,
            self.track_kymo,
            useblit=True,
            button=[3],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )
        self.update_lines()
        self._connect_drag_callback()
        self._connect_line_callback()
        self._select_state({"value": "mode", "old": "", "new": "Track lines"})


class KymoWidgetGreedy(KymoWidget):
    def __init__(
        self,
        kymo,
        channel,
        axis_aspect_ratio=None,
        line_width=None,
        pixel_threshold=None,
        window=4,
        sigma=None,
        vel=0.0,
        diffusion=0.0,
        sigma_cutoff=2.0,
        min_length=3,
        use_widgets=True,
        output_filename="kymotracks.txt",
        slider_ranges={},
        **kwargs,
    ):
        """Create a widget for performing kymotracking.

        Parameters
        ----------
        kymo : lumicks.pylake.Kymo
            Kymograph.
        channel : str
            Kymograph channel to use.
        axis_aspect_ratio : float, optional
            Desired aspect ratio of the viewport. Sometimes kymographs can be very long and thin. This helps you
            visualize them anyway.
        line_width : float, optional
            Expected width of the particles in physical units. Defaults to 4 * pixel size.
        pixel_threshold : float, optional
            Intensity threshold for the pixels. Local maxima above this intensity level will be designated as a line
            origin. Defaults to 98th percentile of the pixel intensities.
        window : int, optional
            Number of kymograph frames in which the particle is allowed to disappear (and still be part of the same
            line). Defaults to 4.
        sigma : float or None, optional
            Uncertainty in the particle position. This parameter will determine whether a peak in the next frame will be
            linked to this one. Increasing this value will make the algorithm tend to allow more positional variation in
            the lines. If none, the algorithm will use half the line width.
        vel : float, optional
            Expected velocity of the traces in the image. This can be used for non-static particles that are expected to
            move at an expected rate (default: 0.0).
        diffusion : float, optional
            Expected diffusion constant (default: 0.0). This parameter will influence whether a peak in the next frame
            will be connected to this one. Increasing this value will make the algorithm allow more positional variation
            in.
        sigma_cutoff : float, optional
            Sets at how many standard deviations from the expected trajectory a particle no longer belongs to this
            trace.
            Lower values result in traces being more stringent in terms of continuing (default: 2.0).
        min_length : int, optional
            Minimum length of a trace. Minimum number of frames a spot has to be detected in to be
            considered. Defaults to 3.
        use_widgets : bool, optional
            Add interactive widgets for interacting with algorithm parameters.
        output_filename : str, optional
            Filename to save to and load from.
        slider_ranges : dict of list, optional
            Dictionary with custom ranges for selected parameter sliders. Ranges should be in the
            following format: (lower bound, upper bound).
            Valid options are: "window", "pixel_threshold", "line_width", "sigma", "min_length" and
            "vel".
        """
        algorithm = track_greedy
        calibrated_kymo_channel = CalibratedKymographChannel.from_kymo(kymo, channel)
        position_scale = calibrated_kymo_channel._pixel_size
        line_width = 4 * position_scale if line_width is None else line_width
        algorithm_parameters = {
            "line_width": line_width,
            "pixel_threshold": np.percentile(calibrated_kymo_channel.data.flatten(), 98)
            if pixel_threshold is None
            else pixel_threshold,
            "window": window,
            "sigma": 0.5 * line_width if sigma is None else sigma,
            "vel": vel,
            "diffusion": diffusion,
            "sigma_cutoff": sigma_cutoff,
        }

        position_scale = calibrated_kymo_channel._pixel_size
        vel_calibration = position_scale / calibrated_kymo_channel.line_time_seconds

        self._slider_ranges = {
            "window": (1, 15),
            "pixel_threshold": (1, np.max(calibrated_kymo_channel.data)),
            "line_width": (0.0, 15.0 * position_scale),
            "sigma": (1.0 * position_scale, 5.0 * position_scale),
            "vel": (-5.0 * vel_calibration, 5.0 * vel_calibration),
            "min_length": (1, 10),
        }
        for key, slider_range in slider_ranges.items():
            if key not in self._slider_ranges:
                raise KeyError(
                    f"Slider range provided for parameter that does not exist ({key}) "
                    f"Valid parameters are: {list(self._slider_ranges.keys())}"
                )

            if len(slider_range) != 2:
                raise ValueError(
                    f"Slider range for parameter {key} should be given as "
                    f"(lower bound, upper bound)."
                )

            if slider_range[1] < slider_range[0]:
                raise ValueError(
                    f"Lower bound should be lower than upper bound for parameter {key}"
                )

            self._slider_ranges[key] = slider_range

        super().__init__(
            kymo,
            channel,
            axis_aspect_ratio,
            min_length,
            use_widgets,
            output_filename,
            algorithm,
            algorithm_parameters,
            min_length_range=self._slider_ranges["min_length"],
            **kwargs,
        )

    def create_algorithm_sliders(self):
        import ipywidgets

        window_slider = self._add_slider(
            "Window",
            "window",
            "How many frames can a line disappear.",
            *self._slider_ranges["window"],
            slider_type=ipywidgets.IntSlider,
        )
        thresh_slider = self._add_slider(
            "Threshold",
            "pixel_threshold",
            "Set the pixel threshold.",
            *self._slider_ranges["pixel_threshold"],
            step_size=1,
            slider_type=ipywidgets.IntSlider,
        )
        line_width_slider = self._add_slider(
            "Line width",
            "line_width",
            "Estimated spot width.",
            *self._slider_ranges["line_width"],
            step_size=1e-3 * self._slider_ranges["line_width"][1],
            slider_type=ipywidgets.FloatSlider,
        )
        sigma_slider = self._add_slider(
            "Sigma",
            "sigma",
            "How much does the line fluctuate?",
            *self._slider_ranges["sigma"],
            step_size=1e-3 * (self._slider_ranges["sigma"][1] - self._slider_ranges["sigma"][0]),
            slider_type=ipywidgets.FloatSlider,
        )
        vel_slider = self._add_slider(
            "Velocity",
            "vel",
            "How fast does the particle move?",
            *self._slider_ranges["vel"],
            step_size=1e-3 * (self._slider_ranges["vel"][1] - self._slider_ranges["vel"][0]),
            slider_type=ipywidgets.FloatSlider,
        )
        return ipywidgets.VBox(
            [thresh_slider, line_width_slider, window_slider, sigma_slider, vel_slider]
        )
