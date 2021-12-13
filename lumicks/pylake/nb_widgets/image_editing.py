import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class ImageStackAxes(Axes):
    def __init__(
        self, *args, image, frame=0, channel="rgb", show_title=True, plot_kwargs={}, **kwargs
    ):
        """Custom axes to handle mouse events for rotating images about a defined tether.

        Parameters
        ----------
        *args
            positional arguments, forwarded to superclass
        image : lk.CorrelatedStack
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
        self.current_image = image

        self.channel = channel
        self.current_frame = frame
        self.num_frames = self.current_image.num_frames
        self.make_title = (
            (lambda: f"{image.name}\n[frame {self.current_frame + 1} / {self.num_frames}]")
            if show_title
            else (lambda: "")
        )

        self.get_figure().canvas.mpl_connect("scroll_event", self.handle_scroll_event)

        self.im = self.imshow(self.get_frame_data(), **plot_kwargs)
        self.set_title(self.make_title())

    def get_frame_data(self):
        return self.current_image._get_frame(self.current_frame)._get_plot_data(self.channel)

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
        self, *args, image, frame=0, channel="rgb", show_title=True, plot_kwargs={}, **kwargs
    ):
        """Custom axes to handle mouse events for rotating images about a defined tether.

        Parameters
        ----------
        *args
            positional arguments, forwarded to superclass
        image : lk.CorrelatedStack
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

    def handle_button_event(self, event):
        """Function to handle mouse click events."""
        if event.inaxes != self:
            return

        # Check if we aren't using a figure widget function like zoom.
        if event.canvas.widgetlock.locked():
            return

        if event.button == 1:
            self.add_point(event.xdata, event.ydata)

    def add_point(self, x, y):
        """Add a point to the tether coordinates; if both ends are defined, rotate the image."""
        self.current_points.append(np.array((x, y)))

        if len(self.current_points) == 2:
            self.current_image = self.current_image.define_tether(*self.current_points)
            temp_tether = np.vstack(self.current_image[0].src._tether.ends)
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


@dataclass
class ImageEditorProjection:
    image: object
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
    def __init__(self, image, frame=0, channel="rgb", show_title=True, **kwargs):
        """Wrapper class to handle interactive tether axes.

        Parameters
        ----------
        image : lk.CorrelatedStack
            image object
        """
        plt.figure()
        self._ax = plt.subplot(
            1, 1, 1, projection=ImageEditorProjection(image, frame, channel, show_title, kwargs)
        )

    @property
    def image(self):
        """Return the edited image object."""
        return self._ax.current_image
