import numpy as np
import warnings


def plot_correlated(
    channel_slice, frame_timestamps, get_plot_data, frame=0, reduce=np.mean, colormap="gray"
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
    frame : int
        Frame to show.
    reduce : callable
        The function which is going to reduce multiple samples into one. The default is
        :func:`numpy.mean`, but :func:`numpy.sum` could also be appropriate for some cases
        e.g. photon counts.
    colormap : str or Colormap
        Colormap used for plotting.
    """
    import matplotlib.pyplot as plt

    downsampled = channel_slice.downsampled_over(frame_timestamps, where="left", reduce=reduce)

    if len(downsampled.timestamps) < len(frame_timestamps):
        warnings.warn("Only subset of time range available for selected channel")

    plot_data = get_plot_data(frame)
    aspect_ratio = plot_data.shape[0] / np.max([plot_data.shape])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plt.figaspect(aspect_ratio / 2))
    t0 = downsampled.timestamps[0]
    t, y = (downsampled.timestamps - t0) / 1e9, downsampled.data
    ax1.step(t, y, where="pre")
    ax2.tick_params(
        axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )
    image_object = ax2.imshow(plot_data, cmap=colormap)
    plt.title(f"Frame {frame}")

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

    def select_frame(event):
        nonlocal poly

        if not event.canvas.widgetlock.locked() and event.inaxes == ax1:
            time = event.xdata * 1e9 + t0
            for img_idx, (start, stop) in enumerate(frame_timestamps):
                if start <= time < stop:
                    plt.title(f"Frame {img_idx}")
                    poly.remove()
                    image_object.set_data(get_plot_data(img_idx))
                    poly = update_position(*frame_timestamps[img_idx])
                    fig.canvas.draw()
                    return

    fig.canvas.mpl_connect("button_press_event", select_frame)
