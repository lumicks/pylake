from lumicks.pylake.kymotracker.detail.trace_line_2d import KymoLine
import numpy as np


def test_kymoline_interpolation():
    time_idx = [1.0, 3.0, 5.0]
    coordinate_idx = [1.0, 3.0, 3.0]
    interpolated = KymoLine(time_idx, coordinate_idx).interpolate()
    np.allclose(interpolated.time_idx, [1.0, 2.0, 3.0, 4.0, 5.0])
    np.allclose(interpolated.coordinate_idx, [1.0, 2.0, 3.0, 3.0, 3.0])
