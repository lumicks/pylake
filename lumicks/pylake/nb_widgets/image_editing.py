from dataclasses import field, dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.widgets import RectangleSelector

from ..kymo import Kymo
from ..image_stack import ImageStack
from ..detail.image import make_image_title


def add_selector(axes, callback, button=None, interactive=True, **kwargs):
    from inspect import signature

    # Remove once matplotlib >= 3.5.0
    props = "props" if "props" in signature(RectangleSelector).parameters else "rectprops"

    kwargs = {
        "button": button,
        "interactive": interactive,
        props: {
            "facecolor": "none",
            "edgecolor": "w",
            "fill": False,
        },
        **kwargs,
    }

    return RectangleSelector(axes, callback, useblit=True, spancoords="data", **kwargs)


class ImageStackAxes(Axes):
    def __init__(
        self, *args, image, frame=0, channel="rgb", show_title=True, plot_kwargs, **kwargs
    ):
        """Custom axes to handle mouse events for rotating images about a defined tether.

        Parameters
        ----------
        *args
            positional arguments, forwarded to superclass
        image : lk.ImageStack
            image object
        frame : int
            initial frame index to plot
        channel : {'rgb', 'red', 'green', 'blue', None}
            Channel to plot for RGB images (None defaults to 'rgb')
            Not used for grayscale images
        show_title : bool
            Controls display of auto-generated plot title
        plot_kwargs : dict
            plotting keyword arguments, forwarded to `Axes.imshow()`
        **kwargs
            keyword arguments, forwarded to superclass
        """
        super().__init__(*args, **kwargs)
        self._current_image = image

        self.channel = channel
        self.current_frame = frame
        self.num_frames = self._current_image.num_frames
        self.make_title = (
            lambda: make_image_title(self._current_image, self.current_frame, show_name=False)
            if show_title
            else (lambda: "")
        )

        self.get_figure().canvas.mpl_connect("scroll_event", self.handle_scroll_event)

        image.plot(frame=frame, channel=channel, axes=self, **plot_kwargs)
        self.im = self.get_images()[-1]
        self.set_title(self.make_title())

    def get_frame_data(self):
        return self._current_image._get_frame(self.current_frame)._get_plot_data(self.channel)

    def update_image(self):
        self.im.set_data(self.get_frame_data())
        self.set_title(self.make_title())

    def handle_scroll_event(self, event):
        if event.inaxes != self:
            return

        if event.button == "up":
            if self.current_frame < self.num_frames - 1:
                self.current_frame += 1
        else:
            if self.current_frame != 0:
                self.current_frame -= 1

        self.update_image()
        self.get_figure().canvas.draw()


class ImageEditorAxes(ImageStackAxes):
    def __init__(
        self, *args, image, frame=0, channel="rgb", show_title=True, plot_kwargs, **kwargs
    ):
        """Custom axes to handle mouse events for rotating images about a defined tether.

        Parameters
        ----------
        *args
            positional arguments, forwarded to superclass
        image : lk.ImageStack
            image object
        frame : int
            initial frame index to plot
        channel : {'rgb', 'red', 'green', 'blue', None}
            Channel to plot for RGB images (None defaults to 'rgb')
            Not used for grayscale images
        show_title : bool
            Controls display of auto-generated plot title
        plot_kwargs : dict
            plotting keyword arguments, forwarded to `Axes.imshow()`
        **kwargs
            keyword arguments, forwarded to superclass
        """
        super().__init__(
            *args,
            image=image,
            frame=frame,
            channel=channel,
            show_title=show_title,
            plot_kwargs=plot_kwargs,
            **kwargs,
        )

        self.current_points = []
        (self._tether_line,) = self.plot([], [], marker="o", mfc="none", c="w", ls="--", lw="0.5")
        self.get_figure().canvas.mpl_connect("button_press_event", self.handle_button_event)

        self.roi_limits = None
        self.selector = add_selector(self, self.handle_crop, button=3, interactive=True)

    def handle_button_event(self, event):
        """Function to handle mouse click events."""
        if event.inaxes != self:
            return

        # Check if we aren't using a figure widget function like zoom.
        if event.canvas.widgetlock.locked():
            return

        if event.button == 1:
            self.add_point(event.xdata, event.ydata)

    def handle_crop(self, event_click, event_release):
        corners = (
            np.sort([event.xdata for event in (event_click, event_release)]),
            np.sort([event.ydata for event in (event_click, event_release)]),
        )
        self.roi_limits = np.hstack(corners)

    def add_point(self, x, y):
        """Add a point to the tether coordinates; if both ends are defined, rotate the image."""
        self.current_points.append(np.array((x, y)))

        if len(self.current_points) == 2:
            self._current_image = self._current_image.define_tether(*self.current_points)
            temp_tether = np.vstack(
                self._current_image[0]._src._tether.ends
                * self._current_image[0]._pixel_calibration_factors
            )
            self.current_points = []
        else:
            temp_tether = np.atleast_2d(np.vstack(self.current_points))

        self.update_plot(temp_tether)

    def update_plot(self, tether_coordinates):
        """Update plot data and re-draw."""
        current_limits = [self.get_xlim(), self.get_ylim()]

        self.update_image()
        self._tether_line.set_data(tether_coordinates[:, 0], tether_coordinates[:, 1])

        for lims, setter in zip(current_limits, [self.set_xlim, self.set_ylim]):
            setter(lims)
        self.get_figure().canvas.draw()

    def get_edited_image(self):
        return (
            self._current_image
            if self.roi_limits is None
            else self._current_image._crop(*self.roi_limits)
        )


@dataclass
class ImageEditorProjection:
    image: ImageStack
    frame: int
    channel: str
    show_title: bool
    plot_kwargs: field(default_factory=dict)

    def _as_mpl_axes(self):
        return ImageEditorAxes, {
            "image": self.image,
            "frame": self.frame,
            "channel": self.channel,
            "show_title": self.show_title,
            "plot_kwargs": self.plot_kwargs,
        }


class ImageEditorWidget:
    """Open a widget to interactively edit the image stack using a tether.

    Actions
    -------
    mouse wheel
        Scroll through frames of the stack.
    left-click
        Define the location of the tether. First click defines the start of the tether
        and second click defines the end. Subsequent clicks will cycle back to re-defining
        the start, etc.
    right-click and drag
        Define the ROI to be cropped.

    Parameters
    ----------
    image : lumicks.pylake.ImageStack
        Image stack object
    frame : int, optional
        Frame index. Default is 0.
    channel : str, optional
        Channel. Default is "rgb".
    show_title : bool, optional
        Show title above the plot. Default is True.
    **kwargs
        Forwarded to plotting function.
    """

    def __init__(self, image, frame=0, channel="rgb", show_title=True, **kwargs):
        """Wrapper class to handle interactive tether axes.

        Parameters
        ----------
        image : lk.ImageStack
            Image stack object
        """
        import matplotlib.pyplot as plt

        plt.figure()
        self._ax = plt.subplot(
            1, 1, 1, projection=ImageEditorProjection(image, frame, channel, show_title, kwargs)
        )

    @property
    def image(self):
        """Return the edited image object."""
        return self._ax.get_edited_image()


class KymoEditorAxes(Axes):
    def __init__(self, *args, kymo, channel="rgb", plot_kwargs, **kwargs):
        """Custom axes to handle mouse events for rotating images about a defined tether.

        Parameters
        ----------
        *args
            positional arguments, forwarded to superclass
        kymo : lk.kymo.Kymo
            kymograph object
        channel : {'rgb', 'red', 'green', 'blue'}
            Color channel to plot
        plot_kwargs : dict
            plotting keyword arguments, forwarded to `Kymo.plot()`
        **kwargs
            keyword arguments, forwarded to superclass
        """
        super().__init__(*args, **kwargs)
        self._kymo = kymo
        self._kymo.plot(channel, axes=self, **plot_kwargs)

        self.time_limits = None
        self.position_limits = None
        self.selector = add_selector(self, self.handle_crop, button=1, interactive=True)

    def handle_crop(self, event_click, event_release):
        self.time_limits = np.sort([event.xdata for event in (event_click, event_release)])
        self.position_limits = np.sort([event.ydata for event in (event_click, event_release)])

    def get_edited_kymo(self):
        if self.time_limits is None and self.position_limits is None:
            return self._kymo

        time_slice = slice(*[f"{lim}s" for lim in self.time_limits])
        kymo = self._kymo[time_slice]
        kymo = kymo.crop_by_distance(*self.position_limits)
        return kymo


@dataclass
class KymoEditorProjection:
    kymo: Kymo
    channel: str
    plot_kwargs: field(default_factory=dict)

    def _as_mpl_axes(self):
        return KymoEditorAxes, {
            "kymo": self.kymo,
            "channel": self.channel,
            "plot_kwargs": self.plot_kwargs,
        }


class KymoEditorWidget:
    def __init__(self, kymo, channel="rgb", tether_length_kbp=None, **kwargs):
        """Wrapper class to handle interactive tether axes.

        Parameters
        ----------
        kymo: lk.kymo.Kymo
            Kymograph object
        channel : {'rgb', 'red', 'green', 'blue'}
            Color channel to plot
        tether_length_kbp : float
            Length of the tether in the cropped region in kilobase pairs.
            If provided, the kymo returned from the `image` property will be automatically
            calibrated to this tether length.
        """
        import matplotlib.pyplot as plt

        kymo._check_is_sliceable()

        plt.figure()
        self._ax = plt.subplot(1, 1, 1, projection=KymoEditorProjection(kymo, channel, kwargs))
        self._tether_length_kbp = tether_length_kbp

    @property
    def kymo(self):
        """Return the cropped kymo object."""
        new_kymo = self._ax.get_edited_kymo()
        if self._tether_length_kbp is not None:
            new_kymo = new_kymo.calibrate_to_kbp(self._tether_length_kbp)
        return new_kymo
