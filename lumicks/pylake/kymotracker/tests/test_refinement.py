from lumicks.pylake.detail.calibrated_images import CalibratedKymographChannel
from lumicks.pylake.kymotracker.kymotracker import refine_lines_centroid
from lumicks.pylake.kymotracker.detail.trace_line_2d import KymoLine
import numpy as np
import pytest


def test_kymoline_interpolation():
    channel = CalibratedKymographChannel(
        "test", np.array([[]]), start=1e9, time_step=1e9, calibration=1
    )
    time_idx = [1.0, 3.0, 5.0]
    coordinate_idx = [1.0, 3.0, 3.0]
    interpolated = KymoLine(time_idx, coordinate_idx, image=channel).interpolate()
    assert np.allclose(interpolated.time_idx, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(interpolated.coordinate_idx, [1.0, 2.0, 3.0, 3.0, 3.0])


def test_refinement_2d():
    time_idx = np.array([1, 2, 3, 4, 5])
    coordinate_idx = np.array([1, 2, 3, 3, 3])

    # Draw image with a deliberate offset
    offset = 2
    data = np.zeros((7, 7))
    data[coordinate_idx + offset, time_idx] = 5
    data[coordinate_idx - 1 + offset, time_idx] = 1
    data[coordinate_idx + 1 + offset, time_idx] = 1
    image = CalibratedKymographChannel.from_array(data)

    line = refine_lines_centroid([KymoLine(time_idx[::2], coordinate_idx[::2], image=image)], 5)[0]
    assert np.allclose(line.time_idx, time_idx)
    assert np.allclose(line.coordinate_idx, coordinate_idx + offset)


@pytest.mark.parametrize("loc", [25.3, 25.5, 26.25, 23.6])
def test_refinement_line(loc, inv_sigma=0.3):
    xx = np.arange(0, 50) - loc
    image = np.exp(-inv_sigma * xx * xx)
    calibrated_image = CalibratedKymographChannel.from_array(np.expand_dims(image, 1))
    line = refine_lines_centroid([KymoLine([0], [25], image=calibrated_image)], 5)[0]
    assert np.allclose(line.coordinate_idx, loc, rtol=1e-2)
