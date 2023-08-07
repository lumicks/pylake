import json
from typing import Union, Iterator

import numpy as np
import tifffile
import numpy.typing as npt

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
        timestamp_ranges = self._tiff_timestamp_ranges()

        # Check length of timestamp ranges, as this is easier than checking the number of frames,
        # which could be an iterator or an np.ndarray.
        if len(timestamp_ranges) == 0:
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
            for frame, timestamp_range in zip(frames, timestamp_ranges):
                tif.write(
                    frame,
                    contiguous=False,  # write tags on each page
                    extratags=extratags(timestamp_range),
                    metadata=None,  # suppress tifffile default ImageDescription tag
                    description=json.dumps(self._tiff_image_metadata(), indent=4),
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

    def _tiff_timestamp_ranges(self) -> Union[list, Iterator]:
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
            First frame in exported video.
        stop_frame : int
            Last frame in exported video.
        fps : int
            Frame rate.
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        scale_bar : lk.ScaleBar
            Scale bar to add to the figure.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`."""
        import matplotlib.pyplot as plt
        from matplotlib import animation

        channels = ("red", "green", "blue", "rgb")
        if channel not in channels:
            raise ValueError(f"Channel should be {', '.join(channels[:-1])} or {channels[-1]}")

        metadata = dict(title=self.name)
        if "ffmpeg" in animation.writers:
            writer = animation.writers["ffmpeg"](fps=fps, metadata=metadata)
        elif "pillow" in animation.writers:
            writer = animation.writers["pillow"](fps=fps, metadata=metadata)
        else:
            raise RuntimeError("You need either ffmpeg or pillow installed to export videos.")

        start_frame = start_frame if start_frame else 0
        stop_frame = stop_frame if stop_frame else self.num_frames

        def plot_func(frame, image_handle):
            return self.plot(
                channel=channel,
                frame=frame,
                image_handle=image_handle,
                adjustment=adjustment,
                scale_bar=scale_bar,
                **kwargs,
            )

        image_handle = None

        def plot(num):
            nonlocal image_handle
            image_handle = plot_func(frame=start_frame + num, image_handle=image_handle)
            return plt.gca().get_children()

        fig = plt.figure()
        fig.patch.set_alpha(1.0)
        line_ani = animation.FuncAnimation(
            fig, plot, stop_frame - start_frame, interval=1, blit=True
        )
        line_ani.save(file_name, writer=writer)
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
