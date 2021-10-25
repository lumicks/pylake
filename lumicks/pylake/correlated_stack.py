import numpy as np
import os
import json
import tifffile
import warnings
from deprecated.sphinx import deprecated
from .detail.widefield import TiffStack


class CorrelatedStack:
    """CorrelatedStack acquired with Bluelake. Bluelake can export stacks of images to various formats. These can be
    opened and correlated to timeline data using CorrelatedStack.

    Parameters
    ----------
    image_name : str
        Filename for the image stack. Typically a TIFF file recorded from a camera in Bluelake.

    align : bool
        If enabled, multi-channel images will be reconstructed from the image alignment metadata
        from Bluelake. The default value is `True`.

    Examples
    --------
    ::

        from lumicks import pylake

        # Loading a stack.
        stack = pylake.CorrelatedStack("example.tiff")

        # Making a plot where force is correlated to images in the stack.
        file = pylake.File("example.h5")
        stack.plot_correlated(file.force1x)

        # Determine the force trace averaged over frame 2...9.
        file.force1x.downsampled_over(stack[2:10].timestamps)
    """

    def __init__(self, image_name, align=True):
        self.src = TiffStack.from_file(image_name, align_requested=align)
        self.name = os.path.splitext(os.path.basename(image_name))[0]
        self.start_idx = 0
        self.stop_idx = self.src.num_frames

    def __getitem__(self, item):
        """All indexing is in frames"""
        if isinstance(item, slice):
            if item.step is not None:
                raise IndexError("Slice steps are not supported")

            start, stop, _ = item.indices(self.num_frames)
            return CorrelatedStack.from_dataset(
                self.src, self.name, self.start_idx + start, self.start_idx + stop
            )
        else:
            item = self.start_idx + item if item >= 0 else self.stop_idx + item
            if item >= self.stop_idx or item < self.start_idx:
                raise IndexError("Index out of bounds")
            return CorrelatedStack.from_dataset(self.src, self.name, item, item + 1)

    def __iter__(self):
        idx = 0
        while idx < self.num_frames:
            yield self._get_frame(idx)
            idx += 1

    @classmethod
    @deprecated(
        reason=("Renamed to `from_dataset()` for consistency with `Kymo` and `Scan`."),
        action="always",
        version="0.10.1",
    )
    def from_data(cls, data, name=None, start_idx=0, stop_idx=None):
        return cls.from_dataset(data, name, start_idx, stop_idx)

    @classmethod
    def from_dataset(cls, data, name=None, start_idx=0, stop_idx=None):
        """Construct CorrelatedStack from image stack object

        Parameters
        ----------
        data : TiffStack
            TiffStack object.
        name : str
            Plot label of the correlated stack
        start_idx : int
            Index at the first frame.
        stop_idx: int
            Index beyond the last frame.
        """
        new_correlated_stack = cls.__new__(cls)
        new_correlated_stack.src = data
        new_correlated_stack.name = name
        new_correlated_stack.start_idx = start_idx
        new_correlated_stack.stop_idx = (
            new_correlated_stack.src.num_frames if stop_idx is None else stop_idx
        )
        return new_correlated_stack

    def crop_by_pixels(self, x_min, x_max, y_min, y_max):
        """Crop the image stack by pixel values.

        Parameters
        ----------
        x_min : int
            minimum x pixel (inclusive)
        x_max : int
            maximum x pixel (exclusive)
        y_min : int
            minimum y pixel (inclusive)
        y_max : int
            maximum y pixel (exclusive)
        """
        data = self.src.with_roi(np.array([x_min, x_max, y_min, y_max]))
        return self.from_dataset(data, self.name, self.start_idx, self.stop_idx)

    def plot(self, frame=0, channel="rgb", show_title=True, **kwargs):
        """Plot image from image stack

        Parameters
        ----------
        frame : int, optional
            Index of the frame to plot.
        channel : 'rgb', 'red', 'green', 'blue', None; optional
            Channel to plot for RGB images (None defaults to 'rgb')
            Not used for grayscale images
        show_title : bool, optional
            Controls display of auto-generated plot title
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        import matplotlib.pyplot as plt

        default_kwargs = dict(cmap="gray", vmax=None)
        kwargs = {**default_kwargs, **kwargs}

        image = self._get_frame(frame)._get_plot_data(channel, vmax=kwargs["vmax"])
        plt.imshow(image, **kwargs)

        if show_title:
            if self.num_frames == 1:
                plt.title(self.name)
            else:
                # display with 1-based index for frames and total frames
                plt.title(f"{self.name} [frame {frame+1}/{self.num_frames}]")

    def _get_frame(self, frame=0):
        if frame >= self.num_frames or frame < 0:
            raise IndexError("Frame index out of range")
        return self.src.get_frame(self.start_idx + frame)

    def plot_correlated(self, channel_slice, frame=0, reduce=np.mean, channel="rgb"):
        """Downsample channel on a frame by frame basis and plot the results. The downsampling function (e.g. np.mean)
        is evaluated for the time between a start and end time of a frame. Note: In environments which support
        interactive figures (e.g. jupyter notebook with ipywidgets or interactive python) this plot will be interactive.

        Parameters
        ----------
        channel_slice : pylake.channel.Slice
            Data slice that we with to downsample.
        frame : int
            Frame to show.
        reduce : callable
            The function which is going to reduce multiple samples into one. The default is
            :func:`numpy.mean`, but :func:`numpy.sum` could also be appropriate for some cases
            e.g. photon counts.
        channel : 'rgb', 'red', 'green', 'blue', None; optional
            Channel to plot for RGB images (None defaults to 'rgb')
            Not used for grayscale images


        Examples
        --------
        ::

            from lumicks import pylake

            file = pylake.File("example.h5")
            stack = pylake.CorrelatedStack("example.tiff")
            stack.plot_correlated(file.force1x, frame=5)
        """
        import matplotlib.pyplot as plt

        downsampled = channel_slice.downsampled_over(self.timestamps, where="left", reduce=reduce)

        if len(downsampled.timestamps) < len(self.timestamps):
            warnings.warn("Only subset of time range available for selected channel")

        fetched_frame = self._get_frame(frame)
        aspect_ratio = fetched_frame.data.shape[0] / np.max([fetched_frame.data.shape])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plt.figaspect(aspect_ratio / 2))
        t0 = downsampled.timestamps[0]
        t, y = (downsampled.timestamps - t0) / 1e9, downsampled.data
        ax1.step(t, y, where="pre")
        ax2.tick_params(
            axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False
        )
        image_object = ax2.imshow(fetched_frame._get_plot_data(channel=channel), cmap="gray")
        plt.title(f"Frame {frame}")

        # Make sure the y-axis limits stay fixed when we add our little indicator rectangle
        y1, y2 = ax1.get_ylim()
        ax1.set_ylim(y1, y2)

        def update_position(new_frame):
            return ax1.fill_between(
                (np.array([new_frame.start, new_frame.stop]) - t0) / 1e9,
                y1,
                y2,
                alpha=0.7,
                color="r",
            )

        poly = update_position(fetched_frame)

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel(downsampled.labels["y"])
        ax1.set_title(downsampled.labels["title"])
        ax1.set_xlim([np.min(t), np.max(t)])

        def select_frame(event):
            nonlocal poly

            if not event.canvas.widgetlock.locked() and event.inaxes == ax1:
                time = event.xdata * 1e9 + t0
                for img_idx in np.arange(0, self.num_frames):
                    current_frame = self._get_frame(img_idx)

                    if current_frame.start <= time < current_frame.stop:
                        plt.title(f"Frame {img_idx}")
                        poly.remove()
                        image_object.set_data(current_frame._get_plot_data(channel=channel))
                        poly = update_position(current_frame)
                        fig.canvas.draw()
                        return

        fig.canvas.mpl_connect("button_press_event", select_frame)

    def export_tiff(self, file_name, roi=None):
        """Export a video of a particular scan plot

        Parameters
        ----------
        file_name : str
            File name to export to.
        roi : list_like
            region of interest in pixel values [xmin, xmax, ymin, ymax]
            *Deprecated since v0.10.1* Instead, use `CorrelatedStack.crop_by_pixels()` to select the ROI before exporting."
        """
        from . import __version__ as version

        def parse_tags(frame):
            # Parse original file tags into list of tuples
            #   [(code, dtype, count, value, writeonce)]
            #   code is defined by the TIFF specification
            #   dtype is defined in tifffile
            # Only tags that are not resolved automatically by `TiffWriter.save()` are needed

            # Orientation, uint16, len, ORIENTATION.TOPLEFT
            orientation = (274, "H", 1, 1)

            # SampleFormat, uint16, len, number of channels
            n_channels = 3 if frame.is_rgb else 1
            sample_format = (339, "H", n_channels, (1,) * n_channels)

            # DateTime, str, len, start:stop
            datetime = frame._page.tags["DateTime"].value
            datetime = (306, "s", len(datetime), datetime)

            return (orientation, sample_format, datetime)

        # crop to ROI if applicable
        if roi is not None:
            warnings.warn(
                (
                    "The `roi` argument is deprecated since v0.10.1 "
                    "Instead, use `CorrelatedStack.crop_by_pixels()` to select the ROI before exporting."
                ),
                DeprecationWarning,
            )
            to_save = self.crop_by_pixels(*roi)
        else:
            to_save = self

        # re-name alignment matrices fields in image description
        # to reflect the fact that the image has already been processed
        description = to_save.src._description.for_export

        # add pylake to Software tag
        software = to_save.src._tags["Software"].value
        if "pylake" not in software:
            software += f", pylake v{version}"

        # write frames sequentially
        with tifffile.TiffWriter(file_name) as tif:
            for frame in to_save:
                tif.save(
                    frame.data,
                    description=description,
                    software=software,
                    metadata=None,  # suppress tifffile default ImageDescription tag
                    contiguous=False,  # needed to write tags on each page
                    extratags=parse_tags(frame),
                    photometric="rgb" if frame.is_rgb else "minisblack",
                )

    @property
    def num_frames(self):
        """Number of frames in the stack."""
        return self.stop_idx - self.start_idx

    @property
    def raw(self):
        """Raw frame data."""
        if self.num_frames > 1:
            return [self._get_frame(idx) for idx in range(self.num_frames)]
        else:
            return self._get_frame(0)

    @property
    def start(self):
        """Starting time stamp of the stack."""
        return self._get_frame(0).start

    @property
    def stop(self):
        """Final time stamp of the stack."""
        return self._get_frame(self.num_frames - 1).stop

    @property
    def timestamps(self):
        """List of time stamps."""
        return [
            (self._get_frame(idx).start, self._get_frame(idx).stop)
            for idx in range(self.num_frames)
        ]
