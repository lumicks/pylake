from ..adjustments import ColorAdjustment
from .timeindex import to_timestamp
import numpy as np
from typing import Union

_FIRST_TIMESTAMP = 1388534400


class VideoExport:
    def export_video(
        self,
        channel,
        file_name,
        *,
        start_frame=None,
        stop_frame=None,
        fps=15,
        adjustment=ColorAdjustment.nothing(),
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
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`."""
        from matplotlib import animation
        import matplotlib.pyplot as plt

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
                **kwargs,
            )

        image_handle = None

        def plot(num):
            nonlocal image_handle
            image_handle = plot_func(frame=start_frame + num, image_handle=image_handle)
            return plt.gca().get_children()

        fig = plt.gcf()
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
