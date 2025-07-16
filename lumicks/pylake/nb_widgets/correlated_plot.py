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
    channel_slice : pylake.channel.Slice | List[pylake.channel.Slice]
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

    channel_slices = channel_slice if isinstance(channel_slice, list) else [channel_slice]

    def downsample_and_validate_dset(dset):
        dset = (
            dset.downsampled_over(frame_timestamps, where="left", reduce=reduce)
            if downsample_to_frames
            else dset[frame_timestamps[0][0] : frame_timestamps[-1][-1]]
        )

        if len(dset) < 2:
            raise ValueError("Channel slice must contain at least two data points.")

        if len(dset.timestamps) < len(frame_timestamps):
            warnings.warn("Only subset of time range available for selected channel")

        return dset

    processed_dsets = [downsample_and_validate_dset(dset) for dset in channel_slices]

    plot_data = get_plot_data(frame)
    aspect_ratio = plot_data.shape[0] / np.max([plot_data.shape])

    aspect_ratio = max(0.2, aspect_ratio)
    if vertical:
        num_plots = 1 + len(processed_dsets)
        fig, axes = plt.subplots(
            num_plots,
            1,
            figsize=figure_scale * plt.figaspect(num_plots * aspect_ratio) * num_plots / 1.5,
            gridspec_kw={"width_ratios": [1]},
        )
        ax_img = axes[0]
        ax_channels = axes[1:]
    else:
        fig, axes = plt.subplots(
            1,
            1 + len(processed_dsets),
            figsize=figure_scale
            * plt.figaspect(aspect_ratio / (len(processed_dsets) * aspect_ratio + 1)),
            gridspec_kw={"width_ratios": [1] * len(processed_dsets) + [1 / aspect_ratio]},
        )
        ax_channels = axes[:-1]
        ax_img = axes[-1]

    t_start = np.min([dset.timestamps[0] for dset in processed_dsets])
    t_stop = np.max([dset.timestamps[-1] for dset in processed_dsets])

    # Find the frame time of the last frame that we still want to plot
    last_dt = (
        np.diff(
            [
                frame_range
                for frame_range in frame_timestamps
                if frame_range[0] >= t_start and frame_range[0] <= t_stop
            ][-1]
        )
        if downsample_to_frames
        else (processed_dsets[0].timestamps[-1] - processed_dsets[0].timestamps[-2])
    )

    def extract_timeseries(processed_channel):
        t, y = (processed_channel.timestamps - t_start) / 1e9, processed_channel.data
        return np.hstack((t, t[-1] + last_dt * 1e-9)), np.hstack((y, y[-1]))

    # We want a constant line from the start of the first frame, to the end. So we plot up to
    # the second point.
    for dset, ax_channel in zip(processed_dsets, ax_channels):
        t, y = extract_timeseries(dset)
        ax_channel.step(t, y, where="post")
        ax_img.tick_params(
            axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False
        )

        ax_channel.set_xlabel("Time [s]")
        ax_channel.set_ylabel(dset.labels["y"])
        ax_channel.set_title(dset.labels["title"])
        ax_channel.set_xlim([0, (t_stop - t_start + last_dt) / 1e9])

    image_object = ax_img.imshow(plot_data, cmap=colormap)
    if post_update:
        post_update(image_object, plot_data)
    ax_img.set_title(title_factory(frame))

    # Make sure the y-axis limits stay fixed when we add our little indicator rectangle
    for ax_channel in ax_channels:
        y1, y2 = ax_channel.get_ylim()
        ax_channel.set_ylim(y1, y2)

    def update_position(start, stop):
        return [
            ax.fill_between(
                (np.array([start, stop]) - t_start) / 1e9,
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                alpha=0.7,
                color="r",
            )
            for ax in ax_channels
        ]

    poly = update_position(*frame_timestamps[frame])

    if vertical:
        # Make sure we don't get a really elongated time plot
        for ax_channel in ax_channels:
            x_lims, y_lims = ax_channel.get_xlim(), ax_channel.get_ylim()
            ax_channel.set_aspect(
                aspect_ratio * abs((x_lims[1] - x_lims[0]) / (y_lims[1] - y_lims[0]))
            )

    # For clicking, we want the region between the start of this frame and the start of the next
    # rather than the actual frame ranges.
    frame_change_positions = np.hstack((np.asarray(frame_timestamps)[:, 0], np.inf))
    frame_change_ranges = np.stack((frame_change_positions[:-1], frame_change_positions[1:])).T

    def update_frame(img_idx):
        nonlocal poly
        ax_img.set_title(title_factory(img_idx))
        for p in poly:
            p.remove()
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
        if not event.canvas.widgetlock.locked() and any(event.inaxes == ax for ax in ax_channels):
            time = event.xdata * 1e9 + t_start
            for img_idx, (start, stop) in enumerate(frame_change_ranges):
                if start <= time < stop:
                    update_frame(img_idx)
                    return

    fig.canvas.mpl_connect("button_press_event", select_frame)
    plt.tight_layout()
    return update_frame
