import pytest
import matplotlib.pyplot as plt
from lumicks.pylake.scalebar import _create_scale_legend


def _validate_elements(ref_elements, item):
    children = item.get_children()
    if children:
        for c in children:
            _validate_elements(ref_elements, c)
    else:
        if ref_elements:
            idx = ref_elements.index(str(item))
            if idx is None:
                raise AssertionError(f"Invalid item found {item}")
            else:
                ref_elements.pop(idx)
        else:
            raise AssertionError("Missing element")


@pytest.mark.parametrize(
    "size_x, size_y, label_x, label_y, loc, color, separation, barwidth, fontsize, reference",
    [
        # fmt:off
        # All entries
        (
            1, 2, "x", "y", "upper right", "white", 2, 2, 16,
            ["Text(0, 0, 'y')", "Text(0, 0, 'x')", "Rectangle(xy=(0, 2), width=1, height=0, angle=0)", "Rectangle(xy=(0, 0), width=0, height=2, angle=0)", "Text(0, 0, 'x')"]
        ),
        # No label for x
        (
            1, 2, None, "y", "upper right", "white", 2, 2, 16,
            ["Text(0, 0, 'y')", "Rectangle(xy=(0, 2), width=1, height=0, angle=0)", "Rectangle(xy=(0, 0), width=0, height=2, angle=0)"]
        ),
        # No label for y
        (
            1, 2, "x", None, "upper right", "white", 2, 2, 16,
            ["Rectangle(xy=(0, 2), width=1, height=0, angle=0)", "Rectangle(xy=(0, 0), width=0, height=2, angle=0)", "Text(0, 0, 'x')"]
        ),
        # No labels
        (
            1, 2, None, None, "upper right", "white", 2, 2, 16,
            ["Rectangle(xy=(0, 2), width=1, height=0, angle=0)", "Rectangle(xy=(0, 0), width=0, height=2, angle=0)"]
        ),
        # Only one axis
        (
            1, 0, "x", "y", "upper right", "white", 2, 2, 16,
            ["Rectangle(xy=(0, 0), width=1, height=0, angle=0)", "Text(0, 0, 'x')"],
        ),
        # Only one axis
        (
            0, 2, "x", "y", "upper right", "white", 2, 2, 16,
            ["Text(0, 0, 'y')", "Rectangle(xy=(0, 0), width=0, height=2, angle=0)"],
        ),
        # fmt:on
    ],
)
def test_scalebar(
    size_x, size_y, label_x, label_y, loc, color, separation, barwidth, fontsize, reference
):
    """We validate that the correct elements exist by checking their string representation"""
    plt.figure()
    axes = plt.gca()
    box = _create_scale_legend(
        axes.transData, size_x, size_y, label_x, label_y, loc, color, separation, barwidth, fontsize
    )

    _validate_elements(reference, box)
    if reference:
        raise AssertionError(f"Not all elements were accounted for: {reference}")
