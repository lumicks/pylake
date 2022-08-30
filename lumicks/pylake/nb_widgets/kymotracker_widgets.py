from dataclasses import dataclass
import inspect
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import warnings
from deprecated.sphinx import deprecated
from matplotlib.widgets import RectangleSelector
from lumicks.pylake.kymotracker.kymotracker import track_greedy
from lumicks.pylake import filter_tracks, refine_tracks_centroid
from lumicks.pylake.nb_widgets.detail.mouse import MouseDragCallback
from lumicks.pylake.kymotracker.kymotrack import KymoTrackGroup, import_kymotrackgroup_from_csv
from lumicks.pylake.nb_widgets.detail.undostack import UndoStack


class KymoWidget:
    def __init__(
        self,
        kymo,
        channel,
        *,
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
        self._axis_aspect_ratio = (
            min(
                axis_aspect_ratio,
                (kymo.line_time_seconds * data.shape[1]) / (kymo.pixelsize[0] * data.shape[0]),
            )
            if axis_aspect_ratio
            else None
        )
        self._tracks_history = UndoStack(KymoTrackGroup([]))
        self._plotted_tracks = []
        self._kymo = kymo
        self._channel = channel
        self._labels = {}
        self._fig = None
        self._axes = None
        self._adding = True
        self._show_tracks = True
        self._output_filename = output_filename

        self._area_selector = None
        self._track_connector = None

        self._algorithm = algorithm
        self._algorithm_parameters = algorithm_parameters

        self._show(use_widgets=use_widgets, **kwargs)

    @property
    @deprecated(
        reason=(
            "This property will be removed in a future release. Use the `tracks` property instead."
        ),
        action="always",
        version="0.13.0",
    )
    def lines(self):
        """Detected tracks."""
        return self.tracks

    @property
    def tracks(self):
        """Detected tracks.

        Returns
        -------
        KymoTrackGroup:
            Collection of detected tracks.
        """
        return self._tracks_history.state

    @lines.setter
    @deprecated(
        reason=(
            "This property will be removed in a future release. Use the `tracks` property instead."
        ),
        action="always",
        version="0.13.0",
    )
    def lines(self, new_lines):
        self.tracks = new_lines

    @tracks.setter
    def tracks(self, new_tracks):
        self._tracks_history.state = new_tracks

    @property
    def _track_width_pixels(self):
        return np.ceil(self._algorithm_parameters["track_width"].value / self._kymo.pixelsize[0])

    def _track_kymo(self, click, release):
        """Handle mouse release event.

        Removes tracks in a region, and traces new ones."""
        p1 = [click.xdata, click.ydata]
        p2 = [release.xdata, release.ydata]

        # Explicit copy to make modifications. Current state pushed to undo stack on assignment.
        tracks = copy(self.tracks)
        tracks.remove_tracks_in_rect([p1, p2])

        if self._adding:
            new_tracks = self._track(rect=[p1, p2])
            tracks.extend(new_tracks)

        self.tracks = tracks
        self._update_tracks()

    def _track_all(self):
        """Track the entire kymograph."""
        self.tracks = self._track()
        self._update_tracks()

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
            self._axes.set_xlim(old_xlims - drag_event.dx)
            canvas.draw_idle()

        MouseDragCallback(self._axes, 1, set_xlim)

    def _get_scale(self):
        """Get scaling of the image axes"""
        return tuple(lims[1] - lims[0] for lims in (self._axes.get_xlim(), self._axes.get_ylim()))

    def _connect_track_callback(self):
        canvas = self._axes.figure.canvas
        cutoff_radius = 0.05  # We use a connection cutoff of 5% of the axis ranges
        clicked_track_info = None
        plotted_track = None
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
                for j, track in enumerate(self.tracks)
            ]
            return np.hstack(nodes).T

        def get_nearest(x, y):
            nonlocal nodes
            ref_position = np.array([x, y])
            squared_dist = np.sum(((ref_position - nodes[:, -2:]) / self._get_scale()) ** 2, 1)
            idx = np.argmin(squared_dist)
            return np.sqrt(squared_dist[idx]), idx

        def initiate_track(event):
            nonlocal nodes, clicked_track_info
            if len(self.tracks) == 0:
                return

            nodes = get_node_info()
            distance, idx = get_nearest(event.x, event.y)

            if distance < cutoff_radius:
                track_index, *track_info = nodes[idx]
                clicked_track_info = [self.tracks[int(track_index)], *track_info]
                return True

        def drag_track(event):
            nonlocal clicked_track_info, plotted_track
            if plotted_track:
                plotted_track.remove()
                plotted_track = None

            track, node_index, *_ = clicked_track_info
            plotted_track, *_ = self._axes.plot(
                [track.seconds[int(node_index)], event.x],
                [track.position[int(node_index)], event.y],
                "r",
            )
            canvas.draw_idle()

        def finalize_track(event):
            nonlocal clicked_track_info, plotted_track
            if plotted_track:
                plotted_track.remove()
                plotted_track = None

            distance, idx = get_nearest(event.x, event.y)
            if distance < cutoff_radius:
                # Explicit copy to make modifications. Current state pushed to undo stack on
                # assignment.
                tracks = copy(self.tracks)

                track_index, *track_info = nodes[idx]
                released_track_info = [self.tracks[int(track_index)], *track_info]

                clicked = [clicked_track_info, released_track_info]
                clicked.sort(key=lambda x: x[2])  # by time
                tracks._merge_tracks(*clicked[0][:2], *clicked[1][:2])

                self.tracks = tracks
                self._update_tracks()

        self._track_connector = MouseDragCallback(
            self._axes,
            3,
            drag_track,
            press_callback=initiate_track,
            release_callback=finalize_track,
        )
        self._track_connector.set_active(False)

    def _update_tracks(self):
        for track in self._plotted_tracks:
            track.remove()
        self._plotted_tracks = []

        if self._show_tracks:
            self._plotted_tracks = [
                self._axes.plot(track.seconds, track.position, color="black", linewidth=5)[0]
                for track in self.tracks
            ]
            self._plotted_tracks.extend(
                [
                    self._axes.plot(track.seconds, track.position, markersize=8)[0]
                    for track in self.tracks
                ]
            )

        self._fig.canvas.draw()

    def _save_from_ui(self):
        try:
            self.save_tracks(
                self._output_filename,
                sampling_width=int(np.ceil(0.5 * self._track_width_pixels)),
            )
            self._set_label("status", f"Saved {self._output_filename}")
        except (RuntimeError, IOError) as exception:
            self._set_label("status", str(exception))

    @deprecated(
        reason=("This property will be removed in a future release. Use `save_tracks()` instead."),
        action="always",
        version="0.13.0",
    )
    def save_lines(self, filename, delimiter=";", sampling_width=None):
        """Export KymoTrackGroup to a csv file."""
        self.save_tracks(filename, delimiter, sampling_width)

    def save_tracks(self, filename, delimiter=";", sampling_width=None):
        """Export KymoTrackGroup to a csv file.

        Parameters
        ----------
        filename : str
            Filename to output kymograph tracks to.
        delimiter : str
            Which delimiter to use in the csv file.
        sampling_width : int or None
            When supplied, this will sample the source image around the kymograph track and export
            the summed intensity with the image. The value indicates the number of pixels in either direction
            to sum over.
        """
        self.tracks.save(filename, delimiter, sampling_width)

    def _load_from_ui(self):
        try:
            self.tracks = import_kymotrackgroup_from_csv(
                self._output_filename, self._kymo, self._channel
            )
            self._update_tracks()
            self._set_label("status", f"Loaded {self._output_filename}")
        except (RuntimeError, IOError) as exception:
            self._set_label("status", str(exception))

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

    def _refine(self):
        if self.tracks:
            self.tracks = refine_tracks_centroid(
                self.tracks, self._algorithm_parameters["track_width"].value
            )
            self._update_tracks()
        else:
            self._set_label(
                "status",
                "You need to track this kymograph or load tracks before you can refine them",
            )

    def _create_algorithm_sliders(self):
        raise NotImplementedError(
            "You should be using a class derived from this class to interact with the "
            "kymotracker algorithm"
        )

    def _set_label(self, key, message):
        if self._labels:
            message = message if key != "warning" else f"<font color='red'>{message}"
            self._labels[key].value = message

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

        self._labels["status"] = ipywidgets.Label(value="")
        self._labels["warning"] = ipywidgets.HTML(value="")

        algorithm_sliders = self._create_algorithm_sliders()

        refine_button = ipywidgets.Button(description="Refine Tracks")
        refine_button.on_click(lambda button: self._refine())

        def set__show_tracks(_show_tracks):
            self._show_tracks = _show_tracks
            if self._fig:
                self._update_tracks()

        _show_tracks_toggle = ipywidgets.interactive(
            set__show_tracks,
            _show_tracks=ipywidgets.ToggleButton(
                description="Show Tracks",
                value=self._show_tracks,
                disabled=False,
                button_style="",
                tooltip="Show Tracks\n\nDisabling this will hide all tracks",
            ),
        )

        all_button = ipywidgets.Button(
            description="Track All", tooltip="Track the entire kymograph with current parameters."
        )
        all_button.on_click(lambda button: self._track_all())

        def undo(button):
            self._tracks_history.undo()
            self._update_tracks()

        def redo(button):
            self._tracks_history.redo()
            self._update_tracks()

        undo_button = ipywidgets.Button(description="Undo", tooltip="Undo")
        undo_button.on_click(undo)
        redo_button = ipywidgets.Button(description="Redo", tooltip="Redo")
        redo_button.on_click(redo)

        load_button = ipywidgets.Button(description="Load")
        load_button.on_click(lambda button: self._load_from_ui())

        save_button = ipywidgets.Button(description="Save")
        save_button.on_click(lambda button: self._save_from_ui())

        def set_fn(value):
            self._output_filename = value.new

        fn_widget = ipywidgets.Text(value=self._output_filename, description="File")
        fn_widget.observe(set_fn, "value")

        self._mode = ipywidgets.RadioButtons(
            options=["Track", "Remove Tracks", "Connect Tracks"],
            disabled=False,
            tooltip="Choose between adding, removing, or connecting tracks",
            layout=ipywidgets.Layout(flex_flow="row"),
        )
        self._mode.observe(self._select_state, "value")

        output = ipywidgets.Output()
        ui = ipywidgets.HBox(
            [
                ipywidgets.VBox([output]),
                ipywidgets.VBox(
                    [
                        all_button,
                        algorithm_sliders,
                        ipywidgets.HBox([refine_button, _show_tracks_toggle]),
                        fn_widget,
                        ipywidgets.HBox([undo_button, redo_button]),
                        ipywidgets.HBox([load_button, save_button]),
                        self._mode,
                        self._labels["status"],
                        self._labels["warning"],
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
        self._track_connector.set_active(False)

        if value["new"] == "Track":
            self._set_label("status", f"Drag right mouse button to track an ROI")
            self._area_selector.set_active(True)
            self._adding = True
        elif value["new"] == "Remove Tracks":
            self._set_label("status", f"Drag right mouse button to remove tracks from an ROI")
            self._area_selector.set_active(True)
            self._adding = False
        elif value["new"] == "Connect Tracks":
            self._set_label("status", f"Drag right mouse button to connect two tracks")
            self._track_connector.set_active(True)

    def _show(self, use_widgets, **kwargs):
        if self._fig:
            plt.close(self._fig)
            self._plotted_tracks = []

        if use_widgets:
            self._create_widgets()
        else:
            self._fig = plt.figure()
            self._axes = self._fig.add_subplot(111)

        self._kymo.plot(channel=self._channel, interpolation="nearest", **kwargs)

        if self._axis_aspect_ratio:
            self._axes.set_xlim(
                [
                    0,
                    self._axis_aspect_ratio
                    * (self._kymo.pixelsize[0] * self._kymo.get_image(self._channel).shape[0]),
                ]
            )

        # Prevents the axes from resetting every time new tracks are drawn
        self._axes.autoscale(enable=False)
        plt.tight_layout()

        self._area_selector = RectangleSelector(
            self._axes,
            self._track_kymo,
            useblit=True,
            button=[3],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )
        self._update_tracks()
        self._connect_drag_callback()
        self._connect_track_callback()
        self._select_state({"value": "mode", "old": "", "new": "Track"})


class KymoWidgetGreedy(KymoWidget):
    def __init__(
        self,
        kymo,
        channel,
        *,
        axis_aspect_ratio=None,
        track_width=None,
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
        track_width : float, optional
            Expected width of the particles in physical units. Defaults to 4 * pixel size.
        line_width : float, optional
            **Deprecated** Forwarded to `track_width`.
        pixel_threshold : float, optional
            Intensity threshold for the pixels. Local maxima above this intensity level will be designated as a track
            origin. Defaults to 98th percentile of the pixel intensities.
        window : int, optional
            Number of kymograph frames in which the particle is allowed to disappear (and still be part of the same
            track). Defaults to 4.
        sigma : float or None, optional
            Uncertainty in the particle position. This parameter will determine whether a peak in the next frame will be
            linked to this one. Increasing this value will make the algorithm tend to allow more positional variation in
            the tracks. If none, the algorithm will use half the track width.
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
            Valid options are: "window", "pixel_threshold", "track_width", "sigma", "min_length" and
            "vel".
        """

        def wrapped_track_greedy(kymo, channel, min_length, **kwargs):
            return filter_tracks(
                track_greedy(kymo, channel, **kwargs),
                min_length,
            )

        algorithm = wrapped_track_greedy
        algorithm_parameters = _get_default_parameters(kymo, channel)

        # TODO: remove line_width argument deprecation path
        if "line_width" in slider_ranges.keys():
            warnings.warn(
                DeprecationWarning(
                    "The argument `slider_ranges['line_width']` is deprecated; use `'track_width'` instead."
                ),
                stacklevel=2,
            )
            slider_ranges["track_width"] = slider_ranges["line_width"]
            slider_ranges.pop("line_width")

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

        # TODO: remove line_width argument deprecation path
        if line_width is not None:
            warnings.warn(
                DeprecationWarning(
                    "The argument `line_width` is deprecated; use `track_width` instead."
                ),
                stacklevel=2,
            )
            parameters["track_width"] = line_width

        for key in keys:
            if parameters[key] is not None:
                algorithm_parameters[key].value = parameters[key]
            if key in slider_ranges:
                algorithm_parameters[key].lower_bound = slider_ranges[key][0]
                algorithm_parameters[key].upper_bound = slider_ranges[key][1]

        super().__init__(
            kymo,
            channel,
            axis_aspect_ratio=axis_aspect_ratio,
            use_widgets=use_widgets,
            output_filename=output_filename,
            algorithm=algorithm,
            algorithm_parameters=algorithm_parameters,
            **kwargs,
        )

    def _create_algorithm_sliders(self):
        import ipywidgets

        slider_box = ipywidgets.VBox(
            [
                self._add_slider(key, parameter)
                for key, parameter in self._algorithm_parameters.items()
                if parameter.ui_visible
            ]
        )

        # callback to show warning if threshold is set too low
        image = self._kymo.get_image(self._channel)
        min_threshold = np.percentile(image, 80)

        threshold_slider_index = list(self._algorithm_parameters.keys()).index("pixel_threshold")
        threshold_slider = slider_box.children[threshold_slider_index].children[0]
        threshold_slider_callback = lambda change: self._set_label(
            "warning",
            ""
            if change["new"] > min_threshold
            else f"Tracking with threshold of {change['new']} may be slow.",
        )
        threshold_slider.observe(
            threshold_slider_callback,
            "value",
            "change",
        )
        threshold_slider_callback({"new": threshold_slider.value})

        return slider_box


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
        "track_width": KymotrackerParameter(
            "Track width",
            "Estimated spot width.",
            "float",
            4 * position_scale,
            *(0.0, 15.0 * position_scale),
            True,
        ),
        "window": KymotrackerParameter(
            "Window", "How many frames can a track disappear.", "int", 4, *(1, 15), True
        ),
        "sigma": KymotrackerParameter(
            "Sigma",
            "How much does the track fluctuate?",
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
