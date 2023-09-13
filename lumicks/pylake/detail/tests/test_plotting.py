from inspect import Parameter, signature

import numpy as np
import pytest
from matplotlib import pyplot as plt

from lumicks.pylake.adjustments import ColorAdjustment
from lumicks.pylake.detail.plotting import get_axes, show_image, parse_color_channel


def test_get_axes():
    fig, (ax1, ax2) = plt.subplots(2)
    ih1 = ax1.imshow(np.empty(shape=(2, 2)))
    ih2 = ax2.imshow(np.empty(shape=(2, 2)))

    ax = get_axes(axes=ax1)
    assert ax is ax1
    ax = get_axes()
    assert ax is ax2
    ax = get_axes(image_handle=ih1)
    assert ax is ax1
    ax = get_axes(image_handle=ih2)
    assert ax is ax2
    ax = get_axes(axes=ax1, image_handle=ih1)
    assert ax is ax1
    with pytest.raises(
        ValueError, match="Supplied image_handle with a different axes than the provided axes"
    ):
        ax = get_axes(axes=ax1, image_handle=ih2)

    plt.close(fig)


def test_parse_color_channel():
    for name, result in zip(("red", "green", "blue"), ("r", "g", "b")):
        assert parse_color_channel(name) == result, f"failed on {name}"

    with pytest.raises(
        ValueError,
        match=(
            "channel must be 'red', 'green', 'blue' or a combination "
            "of 'r', 'g', and/or 'b', got 'violet'."
        ),
    ):
        parse_color_channel("violet")

    for channel in ("r", "g", "b", "rg", "rb", "gb", "rgb"):
        assert parse_color_channel(channel) == channel, f"failed on {channel}"

    with pytest.raises(
        ValueError, match="color channel must be in 'rgb' order; got 'bg', expected 'gb'."
    ):
        parse_color_channel("bg")

    with pytest.warns(
        DeprecationWarning,
        match=(
            "In future versions, the `channel` argument will be restricted to lowercase "
            "letters only. Use 'rgb' instead of 'RGB'."
        ),
    ):
        parse_color_channel("RGB")


def test_show_image():
    im1 = np.empty(shape=(2, 2))
    im2 = np.empty(shape=(3, 3))

    fig, (ax1, ax2) = plt.subplots(2)
    ih1 = ax1.imshow(im1)

    with pytest.raises(
        ValueError, match="Supplied image_handle with a different axes than the provided axes"
    ):
        show_image(im1, image_handle=ih1, axes=ax2)

    # Test if image handle is used for plotting
    ih = show_image(im2, image_handle=ih1)
    assert ih is ih1
    np.testing.assert_equal(ih.get_array(), im2)

    # Test if image handle is used for plotting and axes is ignored (no kwargs forwarded to imshow)
    ih = show_image(im1, image_handle=ih1, axes=ax1, url="NOT_FORWARED_KWARG")
    assert ih is ih1
    np.testing.assert_equal(ih.get_array(), im1)
    assert ih.get_url() is None

    # Test if axes is used for plotting (i.e. new image handle is created)
    ih = show_image(im2, axes=ax1)
    assert ih.axes is ax1
    assert ih is not ih1
    np.testing.assert_equal(ih.get_array(), im2)

    # Test if current axes is used for plotting
    ih = show_image(im1)
    assert ih.axes is ax2
    np.testing.assert_equal(ih.get_array(), im1)

    # Test if kwargs are forwarded to plt.imshow()
    ih = show_image(im1, axes=ax1, url="FORWARDED_KWARG")
    assert ih.get_url() == "FORWARDED_KWARG"

    # Test if ColorAdjustment is applied to image_handle
    ih = show_image(im2, axes=ax1, adjustment=ColorAdjustment(1.5, 2, mode="absolute"), channel="g")
    assert ih.norm.vmin == 1.5
    assert ih.norm.vmax == 2

    plt.close(fig)


def _plot_interface_parameters(image=False, stack=False):
    """Get a list of parameters the signature of a plot function is supposed to have

    Parameters
    ----------
    image : bool
        The plot function is supposed to support plotting of images
    stack : bool
        The plot function is supposed to support selecting frames from image stacks

    Returns
    -------
    List[Tuple]
        A list of (name, kind) of expected parameters
    """
    supported_interfaces = ["general"]
    if image:
        supported_interfaces.append("image")
    if stack:
        supported_interfaces.append("stack")

    arguments = [
        # interface type, (name, parameter kind)
        ["general", ("channel", Parameter.POSITIONAL_OR_KEYWORD)],
        ["stack", ("frame", Parameter.KEYWORD_ONLY)],
        ["image", ("adjustment", Parameter.KEYWORD_ONLY)],
        ["general", ("axes", Parameter.KEYWORD_ONLY)],
        ["image", ("image_handle", Parameter.KEYWORD_ONLY)],
        ["general", ("show_title", Parameter.KEYWORD_ONLY)],
        ["general", ("show_axes", Parameter.KEYWORD_ONLY)],
        ["image", ("scale_bar", Parameter.KEYWORD_ONLY)],
        ["general", ("kwargs", Parameter.VAR_KEYWORD)],
    ]

    return [arg[1] for arg in arguments if arg[0] in supported_interfaces]


def _plot_signature_parameters(cls):
    """Get a list of parameters of the signature of the plot method of a class

    Parameters
    ----------
    cls : class
        The class whose plot method's signature parameters should be retrieved

    Notes
    -----
    The first parameter (i.e. `self`) will be excluded from the returned result

    Returns
    -------
    List[Tuple]
        A list of (name, kind) of signature parameters
    """
    s = signature(cls.plot)
    return [(value.name, value.kind) for value in list(s.parameters.values())[1:]]


def implements_plot_interface(cls, image=False, stack=False):
    """Check if plot method signature implements the expected interface"""
    return _plot_interface_parameters(image=image, stack=stack) == _plot_signature_parameters(cls)


def test_plot_interface_implementations():
    from lumicks.pylake import ImageStack
    from lumicks.pylake.kymo import Kymo
    from lumicks.pylake.scan import Scan
    from lumicks.pylake.point_scan import PointScan

    # Check if classes properly implement the "plotting" interface
    assert implements_plot_interface(ImageStack, image=True, stack=True)
    assert implements_plot_interface(Scan, image=True, stack=True)
    assert implements_plot_interface(Kymo, image=True, stack=False)
    assert implements_plot_interface(PointScan, image=False, stack=False)
