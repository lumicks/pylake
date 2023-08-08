from typing import Union, Optional
from dataclasses import field, dataclass


def _create_scale_legend(
    transform,
    size_x,
    size_y,
    label_x,
    label_y,
    loc,
    color,
    separation,
    barwidth,
    fontsize,
    **kwargs,
):
    """Draws a scale legend for the current axis.

    Parameters
    ----------
    transform : :class:`matplotlib.transforms.CompositeGenericTransform`
        The coordinate frame (typically axes.transData).
    size_x, size_y : float or None
        Length of the scale bars in data units. `0` or `None` omits the scale.
    label_x, label_y : str or None
        Labels for the scale bars. None results in no label on the scale.
    loc : str or int
        Position in the containing axis.  Valid locations are 'upper left', 'upper center',
        'upper right', 'center left', 'center', 'center right', 'lower left', 'lower center',
        'lower right'.
    color : Matplotlib color
        Color to use for the text and labels.
    separation : float
        separation between labels and bars in points.
    bar_width : float or None
        Width of the scale bar(s)
    fontsize : float or None
        Font size to use for the labels.
    **kwargs
        additional arguments passed to :class:`matplotlib.offsetbox.AnchoredOffsetBox`."""
    from matplotlib.patches import Rectangle
    from matplotlib.offsetbox import HPacker, VPacker, TextArea, AuxTransformBox, AnchoredOffsetbox

    bars = AuxTransformBox(transform)
    bar_args = {"ec": color, "lw": barwidth, "fc": None}

    if size_x:
        bars.add_artist(Rectangle(xy=(0, size_y), width=size_x, height=0, **bar_args))

    if size_y:
        bars.add_artist(Rectangle(xy=(0, 0), width=0, height=size_y, **bar_args))

    textprops = {"color": color, "fontsize": fontsize}
    if size_x and label_x:
        xlabel = TextArea(label_x, textprops=textprops)
        bars = VPacker(children=[bars, xlabel], align="center", pad=0, sep=separation)

    if size_y and label_y is not None:
        ylabel = TextArea(label_y, textprops=textprops)

        if size_x and label_x is not None:
            # Add a dummy label with the same text such that the label is properly centered.
            # Without this dummy element, the label ends up in the middle of the full vertical
            # height of the scales and the text underneath.
            dummy = TextArea(label_x, textprops={**textprops, "visible": False})
            ylabel = VPacker(children=[ylabel, dummy], align="center", pad=0, sep=0)

        bars = HPacker(children=[ylabel, bars], align="center", pad=0, sep=separation)

    return AnchoredOffsetbox(loc, child=bars, frameon=False, **kwargs)


@dataclass
class ScaleBar:
    """Draws a scale legend for the current axis.

    Parameters
    ----------
    size_x, size_y : float or None
        Width of the scale bars in data units. `0` or `None` omits the scale.
    label_x, label_y : str or None
        Labels for the scale bars. None results in no label on the scale.
    loc : position in the containing axis
        Valid locations are 'upper left', 'upper center', 'upper right', 'center left',
        'center', 'center right', 'lower left', 'lower center', 'lower right'.
    color : Matplotlib color
        Color to use for the text and labels.
    separation : float
        separation between labels and bars in points.
    bar_width : float or None
        Width of the scale bar(s)
    fontsize : float or None
        Font size to use for the labels.
    **kwargs
        additional arguments passed to :class:`matplotlib.offsetbox.AnchoredOffsetBox`.

    Examples
    --------
    ::

        import lumicks.pylake as lk

        # Loading a kymograph.
        h5_file = pylake.File("example.h5")
        _, kymo = h5_file.kymos.popitem()

        # Show default scale bar
        kymo.plot("green", scale_bar=lk.ScaleBar())
        plt.show()

        # Show scale bar with a scale of 10 seconds on the x axis
        kymo.plot("green", scale_bar=lk.ScaleBar(size_x=10))
        plt.show()
    """

    size_x: Optional[float] = None
    size_y: Optional[float] = None
    label_x: Optional[str] = None
    label_y: Optional[str] = None
    loc: Union[str, int] = "upper right"
    color: str = "white"
    separation: Optional[float] = 2.0
    barwidth: Optional[float] = None
    fontsize: Optional[float] = None
    kwargs: dict = field(default_factory=dict)

    def _attach_scale_bar(self, axis, size_x, size_y, x_unit, y_unit):
        """Attach scale bar to an axis

        axis : :class:`matplotlib.pyplot.axis`
            Axis to attach scale bar to
        size_x, size_y : float
            Length of the scale bars in data units.
            These values are only used if the user did not explicitly set one during initialization.
            Note that `0` or `None` omits the scale.
        x_unit, y_unit : str
            Units of the axis this is being added to.
            This value is only used if the user did not explicitly set one during initialization."""

        size_x = self.size_x if self.size_x is not None else size_x
        size_y = self.size_y if self.size_y is not None else size_y
        scale_bar = _create_scale_legend(
            axis.transData,
            size_x,
            size_y,
            self.label_x if self.label_x is not None else f"{size_x} {x_unit}",
            self.label_y if self.label_y is not None else f"{size_y} {y_unit}",
            self.loc,
            self.color,
            self.separation,
            self.barwidth,
            self.fontsize,
            **self.kwargs,
        )
        axis.add_artist(scale_bar)
