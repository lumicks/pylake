from lumicks.pylake.adjustments import ColorAdjustment


class VideoExport:
    def export_video(
        self,
        channel,
        file_name,
        start_frame=None,
        end_frame=None,
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
        end_frame : int
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
        end_frame = end_frame if end_frame else self.num_frames

        plot_func = lambda frame, image_handle: self.plot(
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
            fig, plot, end_frame - start_frame, interval=1, blit=True
        )
        line_ani.save(file_name, writer=writer)
        plt.close(fig)
