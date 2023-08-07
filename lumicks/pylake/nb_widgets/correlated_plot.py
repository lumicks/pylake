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
    """
    import matplotlib.pyplot as plt

    downsampled = channel_slice.downsampled_over(frame_timestamps, where="left", reduce=reduce)

    if len(downsampled.timestamps) < len(frame_timestamps):
        warnings.warn("Only subset of time range available for selected channel")

    plot_data = get_plot_data(frame)
    aspect_ratio = plot_data.shape[0] / np.max([plot_data.shape])

    aspect_ratio = max(0.2, aspect_ratio)
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=figure_scale * plt.figaspect(aspect_ratio / (aspect_ratio + 1)),
        gridspec_kw={"width_ratios": [1, 1 / aspect_ratio]},
    )
    t0 = downsampled.timestamps[0]
    t, y = downsampled.seconds, downsampled.data
    ax1.step(t, y, where="pre")
    ax2.tick_params(
        axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )
    image_object = ax2.imshow(plot_data, cmap=colormap)
    if post_update:
        post_update(image_object, plot_data)
    ax2.set_title(title_factory(frame))

    # Make sure the y-axis limits stay fixed when we add our little indicator rectangle
    y1, y2 = ax1.get_ylim()
    ax1.set_ylim(y1, y2)

    def update_position(start, stop):
        return ax1.fill_between(
            (np.array([start, stop]) - t0) / 1e9,
            y1,
            y2,
            alpha=0.7,
            color="r",
        )

    poly = update_position(*frame_timestamps[frame])

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(downsampled.labels["y"])
    ax1.set_title(downsampled.labels["title"])
    ax1.set_xlim([np.min(t), np.max(t)])

    # For clicking, we want the region between the start of this frame and the start of the next
    # rather than the actual frame ranges.
    frame_change_positions = np.hstack((np.asarray(frame_timestamps)[:, 0], np.inf))
    frame_change_ranges = np.stack((frame_change_positions[:-1], frame_change_positions[1:])).T

    def select_frame(event):
        nonlocal poly

        if not event.canvas.widgetlock.locked() and event.inaxes == ax1:
            time = event.xdata * 1e9 + t0
            for img_idx, (start, stop) in enumerate(frame_change_ranges):
                if start <= time < stop:
                    ax2.set_title(title_factory(img_idx))
                    poly.remove()
                    img_data = get_plot_data(img_idx)
                    image_object.set_data(img_data)
                    if post_update:
                        post_update(image_object, img_data)
                    poly = update_position(*frame_timestamps[img_idx])
                    fig.canvas.draw()
                    return

    fig.canvas.mpl_connect("button_press_event", select_frame)
    plt.tight_layout()
