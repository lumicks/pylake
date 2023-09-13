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
