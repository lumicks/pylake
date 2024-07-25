import warnings

from ..adjustments import no_adjustment


def get_axes(axes=None, image_handle=None):
    """Return `axes` or the axes of the provided `image_handle` or ensure both axes are the same. If
    neither `axes` nor `image_handle` are provided, fallback to the current `matplotlib` axes"""
    import matplotlib.pyplot as plt

    if axes is None:
        axes = plt.gca() if image_handle is None else image_handle.axes
    if image_handle:
        if axes != image_handle.axes:
            raise ValueError("Supplied image_handle with a different axes than the provided axes")
    return axes


def parse_color_channel(channel):
    """Parse user supplied color channel specification to rgb-like format."""

    if channel in (full_colors := {"red": "r", "green": "g", "blue": "b"}):
        channel = full_colors[channel]

    input_channel = channel
    channel = channel.lower()

    # check all specified components in 'rgb'
    if not set(channel).issubset(set("rgb")):
        raise ValueError(
            "channel must be 'red', 'green', 'blue' or a combination of 'r', 'g', and/or 'b', "
            f"got '{channel}'."
        )

    if input_channel != channel:
        warnings.warn(
            DeprecationWarning(
                "In future versions, the `channel` argument will be restricted to lowercase "
                f"letters only. Use '{channel}' instead of '{input_channel}'."
            )
        )

    # check rgb order
    if channel != (correct_order := "".join(sorted(channel)[::-1])):
        raise ValueError(
            f"color channel must be in 'rgb' order; got '{channel}', expected '{correct_order}'."
        )

    return channel


def show_image(
    image,
    adjustment=no_adjustment,
    channel="rgb",
    image_handle=None,
    axes=None,
    **kwargs,
):
    """Plot image on an image handle or fall back to plot on a provided or the current axes"""
    axes = get_axes(axes=axes, image_handle=image_handle)

    if image_handle:
        # Increase plotting speed by updating the image data in an already existing plot
        image_handle.set_data(image)
    else:
        # Fall back to slower re-plotting with `imshow`.
        image_handle = axes.imshow(image, **kwargs)

    adjustment._update_limits(image_handle, image, channel)

    return image_handle


def _annotate(start_time, annotation, annotation_direction, stop_time, **kwargs):
    """Annotate a region or time point with some text

    Parameters
    ----------
    start_time : float
        Start time in seconds
    annotation : str | None
        String to annotate with
    annotation_direction : "horizontal" | "vertical" | None
        Which direction to show the text when annotating.
    stop_time : float
        Stop time in seconds
    **kwargs :
        Arguments forwarded to `matplotlib.text`.
    """

    import matplotlib.pyplot as plt

    ax = plt.gca()
    annotation_args = {
        "va": "top",
        "y": 0.98,
        "s": annotation,
        "transform": ax.get_xaxis_transform(),
    } | kwargs
    if not annotation_direction:
        annotation_direction = "horizontal" if stop_time else "vertical"

    match annotation_direction:
        case "horizontal":
            ax.text(
                ((start_time + stop_time) / 2 if stop_time else start_time),
                ha="center",
                rotation=0,
                **annotation_args,
            )
        case "vertical":
            if stop_time:
                # Prefer using the stop time, as that puts the text _inside_ the shaded area
                ax.text(stop_time, ha="right", rotation=90, **annotation_args)
            else:
                ax.text(start_time, ha="right", rotation=90, **annotation_args)
        case _:
            raise RuntimeError(
                'Invalid value passed for annotation_direction. Expected "horizontal" or '
                f'"vertical", got {annotation_direction}.'
            )
