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
