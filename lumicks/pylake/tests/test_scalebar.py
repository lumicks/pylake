import json

import numpy as np
import pytest
import matplotlib.pyplot as plt

from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.scalebar import ScaleBar, _create_scale_legend
from lumicks.pylake.image_stack import TiffStack, ImageStack

from .data.mock_confocal import generate_scan
from .data.mock_widefield import MockTiffFile, make_frame_times


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


def validate_args(refs):
    def evaluate_args(*args):
        for val, ref in zip(args[1:], refs):
            assert val == ref

        return _create_scale_legend(*args)

    return evaluate_args


@pytest.mark.parametrize(
    "scale_args, refs, calibrate",
    [
        # fmt:off
        ({}, [60.0, 1.0, "60.0 s", "1.0 μm", "upper right", "white", 2.0, None, None], False),
        ({"label_x": "hi"}, [60.0, 1.0, "hi", "1.0 μm", "upper right", "white", 2.0, None, None], False),
        ({"label_y": "hi"}, [60.0, 1.0, "60.0 s", "hi", "upper right", "white", 2.0, None, None], False),
        ({"size_x": 10}, [10.0, 1.0, "10 s", "1.0 μm", "upper right", "white", 2.0, None, None], False),
        ({"size_y": 10}, [60.0, 10.0, "60.0 s", "10 μm", "upper right", "white", 2.0, None, None], False),
        ({"loc": "lower right"}, [60.0, 1.0, "60.0 s", "1.0 μm", "lower right", "white", 2.0, None, None], False),
        ({"color": "blue"}, [60.0, 1.0, "60.0 s", "1.0 μm", "upper right", "blue", 2.0, None, None], False),
        ({"separation": 5}, [60.0, 1.0, "60.0 s", "1.0 μm", "upper right", "white", 5.0, None, None], False),
        ({"barwidth": 12}, [60.0, 1.0, "60.0 s", "1.0 μm", "upper right", "white", 2.0, 12, None], False),
        ({"fontsize": 16}, [60.0, 1.0, "60.0 s", "1.0 μm", "upper right", "white", 2.0, None, 16], False),
        ({}, [60.0, 1.0, "60.0 s", "1.0 kbp", "upper right", "white", 2.0, None, None], True),
        ({"size_y": 2}, [60.0, 2.0, "60.0 s", "2 kbp", "upper right", "white", 2.0, None, None], True),
        # fmt:on
    ],
)
def test_scalebar_kymo(monkeypatch, scale_args, refs, calibrate):
    """Tests integration with Kymo and whether arguments are appropriately weighted (user arguments
    have higher precedence than defaults from the Kymo"""

    with monkeypatch.context() as m:
        kymo = _kymo_from_array(np.ones((2, 2)), "r", line_time_seconds=5, pixel_size_um=2)

        if calibrate:
            kymo = kymo.calibrate_to_kbp(25)

        m.setattr("lumicks.pylake.scalebar._create_scale_legend", validate_args(refs))
        kymo.plot(channel="red", scale_bar=ScaleBar(**scale_args))


def test_scalebar_integration(monkeypatch):
    """ "For Scan and ImageStack we only test the defaults since we extensively tested Kymo"""
    with monkeypatch.context() as m:
        scan = generate_scan("test", np.random.randint(0, 10, size=(1, 2, 2)), [3, 2])
        refs = [1.0, 1.0, "1.0 μm", "1.0 μm", "upper right", "white", 2.0, None, None]
        m.setattr("lumicks.pylake.scalebar._create_scale_legend", validate_args(refs))
        scan.plot("red", scale_bar=ScaleBar())

    for description, ref_sizes in (
        ({}, [100, 100, "100 px", "100 px"]),
        ({"Pixel calibration (nm/pix)": 500}, [1, 1, r"1 μm", r"1 μm"]),
    ):
        with monkeypatch.context() as m:
            stack = ImageStack.from_dataset(
                TiffStack(
                    [
                        MockTiffFile(
                            data=[np.ones((1, 1))] * 2,
                            times=make_frame_times(2),
                            description=description,
                        )
                    ],
                    align_requested=False,
                )
            )
            refs = [*ref_sizes, "upper right", "white", 2.0, None, None]
            m.setattr("lumicks.pylake.scalebar._create_scale_legend", validate_args(refs))
            stack.plot("red", scale_bar=ScaleBar())
