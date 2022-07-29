import time
from dataclasses import dataclass
import inspect
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from matplotlib.widgets import RectangleSelector
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
        use_widgets,
        output_filename,
        algorithm,
        algorithm_parameters,
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
            This helps you visualize them. The aspect ratio is defined in physical spatial and
            temporal units (rather than pixels).
        use_widgets : bool
            Add interactive widgets for interacting with algorithm parameters.
        output_filename : str
            Filename to save to and load from.
        algorithm : callable
            Kymotracking algorithm used
        algorithm_parameters : dict
            Dictionary of `KymotrackerParameter` instances holding the slider attributes
            and values for the tracking algorithm parameters
        **kwargs
            Extra arguments forwarded to imshow.
        """
        data = kymo.get_image(channel)

        # Forcing the aspect ratio only makes sense when the time axis is longer.
        self.axis_aspect_ratio = (
            min(
                axis_aspect_ratio,
                (kymo.line_time_seconds * data.shape[1]) / (kymo.pixelsize[0] * data.shape[0]),
            )
            if axis_aspect_ratio
            else None
        )
        self._lines_history = UndoStack(KymoLineGroup([]))
        self.plotted_lines = []
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
        self._algorithm_parameters = algorithm_parameters

        self.show(use_widgets=use_widgets, **kwargs)

    @property
    def lines(self):
        return self._lines_history.state

    @lines.setter
    def lines(self, new_lines):
        self._lines_history.state = new_lines

    @property
    def _line_width_pixels(self):
        return np.ceil(self._algorithm_parameters["line_width"].value / self._kymo.pixelsize[0])

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
        return self._algorithm(
            self._kymo,
            self._channel,
            **{key: item.value for key, item in self._algorithm_parameters.items()},
            rect=rect,
        )

    def _connect_drag_callback(self):
        canvas = self._axes.figure.canvas

        def set_xlim(drag_event):
            # Callback for dragging the field of view
            old_xlims = np.array(self._axes.get_xlim())
            self._dx = self._dx + drag_event.dx

            # We don't need to update more than 30 times per second (1/30 = 0.033).
            if abs(time.time() - self._last_update) > 0.033:
                self._axes.set_xlim(old_xlims - self._dx)
                self._dx = 0
                self._last_update = time.time()
                canvas.draw_idle()

        MouseDragCallback(self._axes, 1, set_xlim)

    def _connect_line_callback(self):
        canvas = self._axes.figure.canvas
        visible_range = tuple(
            lims[1] - lims[0] for lims in (self._axes.get_xlim(), self._axes.get_ylim())
        )
        cutoff_radius = 0.05  # We use a connection cutoff of 5% of the axis ranges
        clicked_line_info = None
        plotted_line = None

        def initiate_line(event):
            nonlocal clicked_line_info
            if len(self.lines) == 0:
                return

            line_info = _get_nearest(self.lines, event.x, event.y, visible_range, cutoff_radius)
            if line_info:
                clicked_line_info = line_info
                return True

        def drag_line(event):
            nonlocal clicked_line_info, plotted_line
            if plotted_line:
                plotted_line.remove()
                plotted_line = None

            line = clicked_line_info["line"]
            node_index = clicked_line_info["node_index"]
            plotted_line, *_ = self._axes.plot(
                [line.seconds[int(node_index)], event.x],
                [line.position[int(node_index)], event.y],
                "r",
            )
            canvas.draw_idle()

        def finalize_line(event):
            nonlocal clicked_line_info, plotted_line
            if plotted_line:
                plotted_line.remove()
                plotted_line = None

            line_info = _get_nearest(self.lines, event.x, event.y, visible_range, cutoff_radius)
            if line_info:
                released_line_info = line_info

                # Explicit copy to make modifications. Current state pushed to undo stack on
                # assignment.
                lines = copy(self.lines)

                clicked = [clicked_line_info, released_line_info]
                clicked.sort(key=lambda x: x["coordinates"][0])  # by time

                merge_args = [(click["line"], click["node_index"]) for click in clicked]
                lines._merge_lines(*merge_args[0], *merge_args[1])

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

    def _add_slider(self, name, parameter):
        import ipywidgets

        def set_value(value):
            self._algorithm_parameters[name].value = value

        slider_types = {"int": ipywidgets.IntSlider, "float": ipywidgets.FloatSlider}

        return ipywidgets.interactive(
            set_value,
            value=slider_types[parameter.type](
                description=parameter.name,
                description_tooltip=parameter.description,
                min=parameter.lower_bound,
                max=parameter.upper_bound,
                step=parameter.step_size,
                value=self._algorithm_parameters[name].value,
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
        self._kymo.plot(channel=self._channel, interpolation="nearest", **kwargs)

        if self.axis_aspect_ratio:
            self._axes.set_xlim(
                [
                    0,
                    self.axis_aspect_ratio
                    * (self._kymo.pixelsize[0] * self._kymo.get_image(self._channel).shape[0]),
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


def _get_node_info(lines):
    """Build an array of information on all nodes in all tracked lines.

    Parameters
    ----------
    lines: KymoLineGroup
        Tracked lines.

    Returns
    -------
    np.ndarray
        Array of shape [n, 4] with each row representing information about
        a node in a tracked line corresponding to:
        (track index, node index within track, x-coordinate, y-coordinate)
    """

    nodes = [
        np.vstack(
            (
                np.full(len(track), j),
                np.arange(len(track)),
                track.seconds,
                track.position,
            )
        )
        for j, track in enumerate(lines)
    ]
    return np.hstack(nodes).T


def _get_nearest(lines, x, y, visible_range, cutoff_radius=0.05):
    """Get nearest line to mouse click/release.

    Parameters
    ----------
    lines: KymoLineGroup
        Tracked lines.
    x: float
        Clicked x-coordinate.
    y: float
        Clicked y-coorindate.
    visible_range: tuple
        Scaling of the image axes (x_max-x_min, y_max-y_min).
    cutoff_radius: float
        Maximum distance for a node to be considered clicked, defined as a fraction
        of the axes ranges.

    Returns
    -------
    dict
        line instance, node index, and x-y coordinates for click event
    """
    nodes = _get_node_info(lines)
    ref_position = np.array([x, y])
    squared_dist = np.sum(((ref_position - nodes[:, -2:]) / visible_range) ** 2, 1)
    idx = np.argmin(squared_dist)
    distance = np.sqrt(squared_dist[idx])

    if distance < cutoff_radius:
        line_index, node_index, *coordinates = nodes[idx]
        clicked_line_info = {
            "line": lines[int(line_index)],
            "node_index": node_index,
            "coordinates": coordinates,
        }
        return clicked_line_info
    else:
        return {}


class KymoWidgetGreedy(KymoWidget):
    def __init__(
        self,
        kymo,
        channel,
        axis_aspect_ratio=None,
        line_width=None,
        pixel_threshold=None,
        window=None,
        sigma=None,
        vel=None,
        diffusion=None,
        sigma_cutoff=None,
        min_length=None,
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
            Desired aspect ratio of the viewport. Sometimes kymographs can be very long and thin.
            This helps you visualize them. The aspect ratio is defined in physical spatial and
            temporal units (rather than pixels).
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

        def wrapped_track_greedy(kymo, channel, min_length, **kwargs):
            return filter_lines(
                track_greedy(kymo, channel, **kwargs),
                min_length,
            )

        algorithm = wrapped_track_greedy
        algorithm_parameters = _get_default_parameters(kymo, channel)

        # check slider_ranges entries are valid
        keys = tuple(algorithm_parameters.keys())
        for key, slider_range in slider_ranges.items():
            if key not in keys:
                raise KeyError(
                    f"Slider range provided for parameter that does not exist ({key}) "
                    f"Valid parameters are: {', '.join(keys)}"
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

        # update defaults to user-supplied parameters
        arg_names, _, _, values = inspect.getargvalues(inspect.currentframe())
        parameters = {name: values[name] for name in arg_names if name != "self"}
        for key in keys:
            if parameters[key] is not None:
                algorithm_parameters[key].value = parameters[key]
            if key in slider_ranges:
                algorithm_parameters[key].lower_bound = slider_ranges[key][0]
                algorithm_parameters[key].upper_bound = slider_ranges[key][1]

        super().__init__(
            kymo,
            channel,
            axis_aspect_ratio,
            use_widgets,
            output_filename,
            algorithm,
            algorithm_parameters,
            **kwargs,
        )

    def create_algorithm_sliders(self):
        import ipywidgets

        return ipywidgets.VBox(
            [
                self._add_slider(key, parameter)
                for key, parameter in self._algorithm_parameters.items()
                if parameter.ui_visible
            ]
        )


@dataclass
class KymotrackerParameter:
    from numbers import Number

    name: str
    description: str
    type: str
    value: Number
    lower_bound: Number
    upper_bound: Number
    ui_visible: bool

    def __post_init__(self):
        if self.ui_visible and (self.lower_bound is None or self.upper_bound is None):
            raise ValueError(
                "Lower and upper bounds must be supplied for widget to be set as visible."
            )

    @property
    def step_size(self):
        return 1 if self.type == "int" else (1e-3 * (self.upper_bound - self.lower_bound))


def _get_default_parameters(kymo, channel):
    data = kymo.get_image(channel)
    position_scale = kymo.pixelsize[0]
    vel_calibration = position_scale / kymo.line_time_seconds

    return {
        "pixel_threshold": KymotrackerParameter(
            "Threshold",
            "Set the pixel threshold.",
            "int",
            np.percentile(data.flatten(), 98),
            *(1, np.max(data)),
            True,
        ),
        "line_width": KymotrackerParameter(
            "Line width",
            "Estimated spot width.",
            "float",
            4 * position_scale,
            *(0.0, 15.0 * position_scale),
            True,
        ),
        "window": KymotrackerParameter(
            "Window", "How many frames can a line disappear.", "int", 4, *(1, 15), True
        ),
        "sigma": KymotrackerParameter(
            "Sigma",
            "How much does the line fluctuate?",
            "float",
            2 * position_scale,
            *(1.0 * position_scale, 5.0 * position_scale),
            True,
        ),
        "vel": KymotrackerParameter(
            "Velocity",
            "How fast does the particle move?",
            "float",
            0.0,
            *(-5.0 * vel_calibration, 5.0 * vel_calibration),
            True,
        ),
        "diffusion": KymotrackerParameter(
            "Diffusion", "Expected diffusion constant.", "float", 0.0, *(None, None), False
        ),
        "sigma_cutoff": KymotrackerParameter(
            "Sigma cutoff",
            "Number of standard deviations from the expected trajectory a particle no longer belongs to a trace.",
            "float",
            2.0,
            *(None, None),
            False,
        ),
        "min_length": KymotrackerParameter(
            "Min length",
            "Minimum number of frames a spot has to be detected in to be considered.",
            "int",
            3,
            *(1, 10),
            True,
        ),
    }
