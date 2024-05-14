import json
from typing import Union, Iterator

import numpy as np
import tifffile
import numpy.typing as npt

from .plotting import parse_color_channel
from .timeindex import to_timestamp
from ..adjustments import no_adjustment

_FIRST_TIMESTAMP = 1388534400


class TiffExport:
    def export_tiff(self, filename, *, dtype=None, clip=False):
        """Export frames to a TIFF image

        Parameters
        ----------
        filename : str | os.PathLike
            The name of the TIFF file where the image will be saved.
        dtype : np.dtype
            The data type of a single color channel in the resulting image.
        clip : bool
            If enabled, the image data will be clipped to fit into the desired `dtype`. This option
            is disabled by default: an error will be raise if the data does not fit.
        """
        # If the exported tiff should be cast to `dtype`, get the full image stack to later safely
        # cast it. Otherwise, try to get an iterator, to save memory.
        frames = self._tiff_frames(iterator=dtype is None)
        frame_timestamp_ranges = self._tiff_timestamp_ranges(include_dead_time=True)
        frame_exposure_ranges = self._tiff_timestamp_ranges(include_dead_time=False)

        # Check length of timestamp ranges, as this is easier than checking the number of frames,
        # which could be an iterator or an np.ndarray.
        if len(frame_timestamp_ranges) == 0:
            raise RuntimeError("Can't export TIFF if there are no images.")

        # Cast frames if a specific dtype is requested.
        def cast_image(image, dtype=np.float32, clip=False):
            # Check if requested dtype can fit image values without an overflow
            info = np.finfo(dtype) if np.dtype(dtype).kind == "f" else np.iinfo(dtype)
            if np.min(image) < info.min or np.max(image) > info.max:
                if clip:
                    image = np.clip(image, info.min, info.max)
                else:
                    raise RuntimeError(
                        f"Can't safely export image with `dtype={dtype.__name__}` channels. "
                        f"Switch to a larger `dtype` in order to safely store everything or pass "
                        f"`clip=True` to clip the data."
                    )

            return image.astype(dtype)

        frames = cast_image(frames, dtype=dtype, clip=clip) if dtype else frames

        def extratags(timestamp_range):
            """Create the extratags tuple used for the `TiffWriter.write()` method

            Notes
            -----
            See TIFF specification for the definition of tags (or fields)
            """
            # DateTime, str, len, start:stop
            datetime = f"{timestamp_range[0]}:{timestamp_range[1]}"
            datetime = (306, "s", len(datetime), datetime)
            return (datetime,)

        # Save the tiff file page by page
        with tifffile.TiffWriter(filename) as tif:
            metadata = self._tiff_image_metadata()
            exposure_times = (
                np.atleast_1d(np.diff(np.vstack(frame_exposure_ranges), axis=1).squeeze()) * 1e-6
            )
            for frame, timestamp_range, exposure_time in zip(
                frames, frame_timestamp_ranges, exposure_times
            ):
                metadata["Exposure time (ms)"] = exposure_time
                tif.write(
                    frame,
                    contiguous=False,  # write tags on each page
                    extratags=extratags(timestamp_range),
                    metadata=None,  # suppress tifffile default ImageDescription tag
                    description=json.dumps(metadata, indent=4),
                    **self._tiff_writer_kwargs(),
                )

    def _tiff_frames(self, iterator=False) -> Union[npt.ArrayLike, Iterator]:
        """Create frames of TIFFs used by `export_tiff()`."""
        raise NotImplementedError(
            f"`{self.__module__}.{self.__class__.__name__}` does not implement `_tiff_frames()`."
        )

    def _tiff_image_metadata(self) -> dict:
        """Create metadata stored in the ImageDescription field of TIFFs used by `export_tiff()`."""
        raise NotImplementedError(
            f"`{self.__module__}.{self.__class__.__name__}` does not implement `_tiff_image_metadata()`."
        )

    def _tiff_timestamp_ranges(self, include_dead_time) -> Union[list, Iterator]:
        """Create Timestamp ranges for DateTime field of TIFFs used by `export_tiff()`."""
        raise NotImplementedError(
            f"`{self.__module__}.{self.__class__.__name__}` does not implement `_tiff_timestamp_ranges()`."
        )

    def _tiff_writer_kwargs(self) -> dict:
        """Create keyword arguments used for `TiffWriter.write()` in `self.export_tiff()`."""
        raise NotImplementedError(
            f"`{self.__module__}.{self.__class__.__name__}` does not implement `_tiff_writer_kwargs()`."
        )


class VideoExport:
    def export_video(
        self,
        channel,
        file_name,
        *,
        start_frame=None,
        stop_frame=None,
        fps=15,
        adjustment=no_adjustment,
        scale_bar=None,
        channel_slice=None,
        vertical=True,
        downsample_to_frames=True,
        **kwargs,
    ):
        """Export a video

        Parameters
        ----------
        channel : str
            Color channel(s) to use "red", "green", "blue" or "rgb".
        file_name : str
            File name to export to.
        start_frame : int
            First frame in exported video (starts at zero).
        stop_frame : int
            Stop frame in exported video. Note that this frame is no longer included.
        fps : int
            Frame rate.
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        scale_bar : lk.ScaleBar
            Scale bar to add to the figure.
        channel_slice : lk.Slice, optional
            When specified, we export a video correlated to channel data
        vertical : bool, optional
            Render with the plots vertically aligned (default: True).
        downsample_to_frames : bool, optional
            Downsample the channel data over frame timestamp ranges (default: True).
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.

        Examples
        --------
        ::

            import lumicks.pylake as lk

            # Note that the examples are shown for an ImageStack, but the API for Scan stacks is
            # identical.
            imgs = lk.ImageStack("stack.tiff")
            imgs.plot_correlated(f.force1x, frame=5, vertical=False)

            # Perform a basic export of the full stack
            imgs.export_video("rgb", "test.gif")

            # Export the first 10 frames
            imgs[:10].export_video("rgb", "test.gif")

            # Export a cropped video (cropping from pixel 10 to 50 along each axis)
            imgs[:, 10:50, 10:50].export_video("rgb", "test.gif")

            # Export with a color adjustment using percentile based adjustment.
            imgs.export_video(
                "rgb", "test.gif", adjustment=lk.ColorAdjustment(2, 98, mode="percentile"
            )

            # Export a gif at 50 fps
            imgs.export_video("rgb", "test.gif", fps=50)

            # We can also export with correlated channel data
            h5_file = lk.File("stack.h5")

            # Export video with correlated force data. A marker will indicate the frame.
            imgs.export_video("rgb", "test.gif", channel_slice=h5_file.force1x)

            # If you want the subplots side by side, pass the extra argument `vertical=False`.
            imgs.export_video("rgb", "test.gif", channel_slice=h5_file.force1x, vertical=False)

            # Export to a mp4 file, note that this needs ffmpeg to be installed. See:
            # https://lumicks-pylake.readthedocs.io/en/latest/install.html#optional-dependencies
            # for more information.
            imgs.export_video("rgb", "test.mp4")
        """
        import matplotlib.pyplot as plt
        from matplotlib import animation

        channel = parse_color_channel(channel)

        metadata = dict(title=self.name)
        if "ffmpeg" in animation.writers:
            writer = animation.writers["ffmpeg"](fps=fps, metadata=metadata)
        elif "pillow" in animation.writers:
            writer = animation.writers["pillow"](fps=fps, metadata=metadata)
        else:
            raise RuntimeError("You need either ffmpeg or pillow installed to export videos.")

        start_frame = start_frame if start_frame is not None else 0
        stop_frame = stop_frame if stop_frame is not None else self.num_frames

        shared_args = {"channel": channel, "adjustment": adjustment}
        if channel_slice:
            set_frame = self.plot_correlated(
                **shared_args,
                frame=start_frame,
                channel_slice=channel_slice,
                vertical=vertical,
                return_frame_setter=True,
                downsample_to_frames=downsample_to_frames,
            )
            fig = plt.gcf()  # plot_correlated makes its own plot
            fig.patch.set_alpha(1.0)  # Circumvents grainy rendering

            def plot(frame):
                set_frame(frame + start_frame)
                artists = []
                for ax in fig.get_children():
                    artists = ax.get_children()
                    if artists:
                        artists.extend(artists)

                return artists

        else:
            fig = plt.figure()
            fig.patch.set_alpha(1.0)  # Circumvents grainy rendering
            image_handle = None

            def plot(frame):
                nonlocal image_handle
                image_handle = self.plot(
                    **shared_args,
                    frame=frame + start_frame,
                    image_handle=image_handle,
                    scale_bar=scale_bar,
                    **kwargs,
                )
                return plt.gca().get_children()

        # Don't store the FuncAnimation in a variable as this leads to mpl attempting to remove
        # a callback that doesn't exist on plt.close(fig) when using the jupyter notebook backend.
        animation.FuncAnimation(fig, plot, stop_frame - start_frame, interval=1, blit=True).save(
            file_name,
            writer=writer,
        )
        plt.close(fig)


class FrameIndex:
    def _time_to_frame_index(self, time, is_start=False) -> Union[int, None]:
        """Convert timestamps, time strings or frame indices to frame indices

        Parameters
        ----------
        time : str or int or None
            Time expressed in either string format (e.g. "5s"), timestamp in nanoseconds or frame.
        is_start : bool
            Whether the input time refers to the start or stop timestamp of the searched frame
        """
        if time is None:
            return time

        # Decode timestamp from string
        time = to_timestamp(time, self.start, self.stop) if isinstance(time, str) else time

        try:
            if time < _FIRST_TIMESTAMP:
                return time  # It's already a frame index
            else:
                ts = np.asarray(self.frame_timestamp_ranges())[:, 0 if is_start else 1]
                return np.searchsorted(ts, time)
        except TypeError:
            raise IndexError(f"Slicing by {type(time).__name__} is not supported.")
