import warnings

import numpy as np


def plot_correlated(
    channel_slice,
    frame_timestamps,
    get_plot_data,
    title_factory,
    frame=0,
    reduce=np.mean,
    colormap="gray",
    figure_scale=0.75,
    post_update=None,
    *,
    vertical=False,
    downsample_to_frames=True,
):
    """Downsample channel on a frame by frame basis and plot the results.

    Parameters
    ----------
    channel_slice : pylake.channel.Slice
        Data slice that we with to downsample.
    frame_timestamps : list of tuple
        List of tuples with start and stop timestamps of each frame.
    get_plot_data : callable
        Function that will return the plotdata for a frame.
    title_factory : callable
        Function to generate title for the image plot.
    frame : int
        Frame to show.
    reduce : callable
        The function which is going to reduce multiple samples into one. The default is
        :func:`numpy.mean`, but :func:`numpy.sum` could also be appropriate for some cases
        e.g. photon counts.
    colormap : str or Colormap
        Colormap used for plotting.
    figure_scale : float
        Scaling of the figure width and height. Values greater than one increase the size of the
        figure.
    post_update : callable
        Function that will be called with the imshow handle and image data after the image data has
        been updated.
    vertical : bool
        Whether plots should be aligned vertically.
    return_handle : bool
        Whether to return a handle to the update function.
    downsample_to_frames : bool
        Downsample the channel data over frame timestamp ranges (default: True).
    """
    import matplotlib.pyplot as plt

    processed_channel = (
        channel_slice.downsampled_over(frame_timestamps, where="left", reduce=reduce)
        if downsample_to_frames
        else channel_slice[frame_timestamps[0][0] : frame_timestamps[-1][-1]]
    )

    if len(processed_channel) < 2:
        raise ValueError("Channel slice must contain at least two data points.")

    if len(processed_channel.timestamps) < len(frame_timestamps):
        warnings.warn("Only subset of time range available for selected channel")

    plot_data = get_plot_data(frame)
    aspect_ratio = plot_data.shape[0] / np.max([plot_data.shape])

    aspect_ratio = max(0.2, aspect_ratio)
    if vertical:
        fig, (ax_img, ax_channel) = plt.subplots(2, 1)  # different axis order is on purpose
    else:
        fig, (ax_channel, ax_img) = plt.subplots(
            1,
            2,
            figsize=figure_scale * plt.figaspect(aspect_ratio / (aspect_ratio + 1)),
            gridspec_kw={"width_ratios": [1, 1 / aspect_ratio]},
        )

    t0 = processed_channel.timestamps[0]
    t, y = processed_channel.seconds, processed_channel.data

    # We explicitly append the last frame time to make sure that it still shows up
    last_dt = (
        np.diff(
            [
                frame_range
                for frame_range in frame_timestamps
                if frame_range[0] >= channel_slice.start and frame_range[1] <= channel_slice.stop
            ][-1]
        )
        if downsample_to_frames
        else (processed_channel.timestamps[-1] - processed_channel.timestamps[-2])
    )
    t = np.hstack((t, t[-1] + last_dt * 1e-9))
    y = np.hstack((y, y[-1]))

    # We want a constant line from the start of the first frame, to the end. So we plot up to
    # the second point.
    ax_channel.step(t, y, where="post")
    ax_img.tick_params(
        axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )
    image_object = ax_img.imshow(plot_data, cmap=colormap)
    if post_update:
        post_update(image_object, plot_data)
    ax_img.set_title(title_factory(frame))

    # Make sure the y-axis limits stay fixed when we add our little indicator rectangle
    y1, y2 = ax_channel.get_ylim()
    ax_channel.set_ylim(y1, y2)

    def update_position(start, stop):
        return ax_channel.fill_between(
            (np.array([start, stop]) - t0) / 1e9,
            y1,
            y2,
            alpha=0.7,
            color="r",
        )

    poly = update_position(*frame_timestamps[frame])

    ax_channel.set_xlabel("Time [s]")
    ax_channel.set_ylabel(processed_channel.labels["y"])
    ax_channel.set_title(processed_channel.labels["title"])
    ax_channel.set_xlim([np.min(t), np.max(t)])

    if vertical:
        # Make sure we don't get a really elongated time plot
        x_lims, y_lims = ax_channel.get_xlim(), ax_channel.get_ylim()
        ax_channel.set_aspect(aspect_ratio * abs((x_lims[1] - x_lims[0]) / (y_lims[1] - y_lims[0])))

    # For clicking, we want the region between the start of this frame and the start of the next
    # rather than the actual frame ranges.
    frame_change_positions = np.hstack((np.asarray(frame_timestamps)[:, 0], np.inf))
    frame_change_ranges = np.stack((frame_change_positions[:-1], frame_change_positions[1:])).T

    def update_frame(img_idx):
        nonlocal poly
        ax_img.set_title(title_factory(img_idx))
        poly.remove()
        img_data = get_plot_data(img_idx)
        image_object.set_data(img_data)

        if post_update:
            post_update(image_object, img_data)
        poly = update_position(*frame_timestamps[img_idx])

        # fig.canvas.draw() is needed to refresh in interactive backends, but when exporting to a
        # movie, it complains about the manager property being `None`. This workaround makes it
        # work in both situations.
        if fig.canvas.manager:
            fig.canvas.draw()

    def select_frame(event):
        nonlocal poly

        if not event.canvas.widgetlock.locked() and event.inaxes == ax_channel:
            time = event.xdata * 1e9 + t0
            for img_idx, (start, stop) in enumerate(frame_change_ranges):
                if start <= time < stop:
                    update_frame(img_idx)
                    return

    fig.canvas.mpl_connect("button_press_event", select_frame)
    plt.tight_layout()
    return update_frame
