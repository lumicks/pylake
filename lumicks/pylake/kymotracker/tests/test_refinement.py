from lumicks.pylake.kymotracker.detail.calibrated_images import CalibratedKymographChannel
from lumicks.pylake.kymotracker.kymotracker import refine_lines_centroid, refine_lines_gaussian
from lumicks.pylake.kymotracker.kymoline import KymoLine
import numpy as np
import pytest


def test_kymoline_interpolation():
    time_idx = np.array([1.0, 3.0, 5.0])
    coordinate_idx = np.array([1.0, 3.0, 3.0])
    kymoline = KymoLine(time_idx, coordinate_idx, [])
    interpolated = kymoline.interpolate()
    assert np.allclose(interpolated.time_idx, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(interpolated.coordinate_idx, [1.0, 2.0, 3.0, 3.0, 3.0])

    # Test whether concatenation still works after interpolation
    assert np.allclose(
        (interpolated + kymoline).time_idx, [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 3.0, 5.0]
    )
    assert np.allclose(
        (interpolated + kymoline).coordinate_idx, [1.0, 2.0, 3.0, 3.0, 3.0, 1.0, 3.0, 3.0]
    )


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

    line = KymoLine(time_idx[::2], coordinate_idx[::2], image=image)
    refined_line = refine_lines_centroid([line], 5)[0]
    assert np.allclose(refined_line.time_idx, time_idx)
    assert np.allclose(refined_line.coordinate_idx, coordinate_idx + offset)

    # Test whether concatenation still works after refinement
    assert np.allclose((refined_line + line).time_idx, np.hstack((time_idx, time_idx[::2])))
    assert np.allclose(
        (refined_line + line).coordinate_idx,
        np.hstack((coordinate_idx + offset, coordinate_idx[::2])),
    )


@pytest.mark.parametrize("loc", [25.3, 25.5, 26.25, 23.6])
def test_refinement_line(loc, inv_sigma=0.3):
    xx = np.arange(0, 50) - loc
    image = np.exp(-inv_sigma * xx * xx)
    calibrated_image = CalibratedKymographChannel.from_array(np.expand_dims(image, 1))
    line = refine_lines_centroid([KymoLine([0], [25], image=calibrated_image)], 5)[0]
    assert np.allclose(line.coordinate_idx, loc, rtol=1e-2)


def test_refine_gaussian(high_intensity):
    position, pixel_size, line, photon_count, true_params, image_params = high_intensity
    n_frames = photon_count.shape[1]
    channel = CalibratedKymographChannel.from_array(photon_count, pixel_size=0.1)

    multipliers = (0.99, 0.9, 0.8)
    init_tracks = []
    for m in multipliers:
        init_center = true_params[1] * m
        init_center_idx = init_center / channel._pixel_size
        init_tracks.append(KymoLine(np.arange(n_frames), np.full(n_frames, init_center_idx), channel))

    results = ([2.604524,   2.59820109, 2.5995581 ],
               [2.61053829, 2.58411975, 2.59177182],
               [2.59546472, 2.59955604, 2.59679171])
    with pytest.warns(UserWarning):
        fitted = refine_lines_gaussian(init_tracks)
    for line, result in zip(fitted, results):
        assert np.allclose(result, line.position)
