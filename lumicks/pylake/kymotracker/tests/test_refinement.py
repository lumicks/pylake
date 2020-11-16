from lumicks.pylake.kymotracker.kymotracker import refine_lines_centroid
from lumicks.pylake.kymotracker.detail.trace_line_2d import KymoLine
import numpy as np
import pytest


def test_kymoline_interpolation():
    time_idx = [1.0, 3.0, 5.0]
    coordinate_idx = [1.0, 3.0, 3.0]
    interpolated = KymoLine(time_idx, coordinate_idx).interpolate()
    assert np.allclose(interpolated.time_idx, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(interpolated.coordinate_idx, [1.0, 2.0, 3.0, 3.0, 3.0])


def test_refinement_2d():
    time_idx = np.array([1, 2, 3, 4, 5])
    coordinate_idx = np.array([1, 2, 3, 3, 3])

    # Draw image with a deliberate offset
    offset = 2
    image = np.zeros((7, 7))
    image[coordinate_idx + offset, time_idx] = 5
    image[coordinate_idx - 1 + offset, time_idx] = 1
    image[coordinate_idx + 1 + offset, time_idx] = 1

    line = refine_lines_centroid([KymoLine(time_idx[::2], coordinate_idx[::2], image_data=image)], 5)[0]
    assert np.allclose(line.time_idx, time_idx)
    assert np.allclose(line.coordinate_idx, coordinate_idx + offset)


@pytest.mark.parametrize("loc", [25.3, 25.5, 26.25, 23.6])
def test_refinement_line(loc, inv_sigma=0.3):
    xx = np.arange(0, 50) - loc
    image = np.exp(-inv_sigma * xx * xx)
    line = refine_lines_centroid([KymoLine([0], [25], image_data=np.expand_dims(image, 1))], 5)[0]
    assert np.allclose(line.coordinate_idx, loc, rtol=1e-2)
