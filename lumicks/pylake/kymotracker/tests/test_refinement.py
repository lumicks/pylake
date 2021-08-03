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
    np.testing.assert_allclose(interpolated.time_idx, [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(interpolated.coordinate_idx, [1.0, 2.0, 3.0, 3.0, 3.0])

    # Test whether concatenation still works after interpolation
    np.testing.assert_allclose(
        (interpolated + kymoline).time_idx, [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 3.0, 5.0]
    )
    np.testing.assert_allclose(
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
    np.testing.assert_allclose(refined_line.time_idx, time_idx)
    np.testing.assert_allclose(refined_line.coordinate_idx, coordinate_idx + offset)

    # Test whether concatenation still works after refinement
    np.testing.assert_allclose((refined_line + line).time_idx, np.hstack((time_idx, time_idx[::2])))
    np.testing.assert_allclose(
        (refined_line + line).coordinate_idx,
        np.hstack((coordinate_idx + offset, coordinate_idx[::2])),
    )


@pytest.mark.parametrize("loc", [25.3, 25.5, 26.25, 23.6])
def test_refinement_line(loc, inv_sigma=0.3):
    xx = np.arange(0, 50) - loc
    image = np.exp(-inv_sigma * xx * xx)
    calibrated_image = CalibratedKymographChannel.from_array(np.expand_dims(image, 1))
    line = refine_lines_centroid([KymoLine([0], [25], image=calibrated_image)], 5)[0]
    np.testing.assert_allclose(line.coordinate_idx, loc, rtol=1e-2)


def test_gaussian_refinement(kymogroups_2lines):
    lines, gapped_lines, mixed_lines = kymogroups_2lines

    # full data, no overlap
    refined = refine_lines_gaussian(lines, window=3, refine_missing_frames=True,
                                    overlap_strategy="ignore")
    assert np.allclose(refined[0].position, [3.54449334, 3.52869383, 3.51225048, 3.38714712, 3.48588412])
    assert np.allclose(refined[1].position, [4.96700319, 4.99771575, 5.04086914, 5.0066495,  4.99092852])

    # initial guess for sigma
    refined = refine_lines_gaussian(lines, window=3, refine_missing_frames=True,
                                    overlap_strategy="ignore", initial_sigma=0.250)
    assert np.allclose(refined[0].position, [3.54796251, 3.52869378, 3.51225141, 3.4718899, 3.48588423])
    assert np.allclose(refined[1].position, [4.96700218, 4.99771571, 5.04086917, 5.00664717, 4.9909296])

    # all frames overlap, therefore skipped and result is empty
    with pytest.warns(UserWarning):
        refined = refine_lines_gaussian(lines, window=10, refine_missing_frames=True,
                                        overlap_strategy="skip")
    assert len(refined) == 0

    # invalid overlap strategy
    with pytest.raises(AssertionError):
        refined = refine_lines_gaussian(lines, window=3, refine_missing_frames=True,
                                        overlap_strategy="something")

    # gapped data, fill in missing frames
    refined = refine_lines_gaussian(gapped_lines, window=3, refine_missing_frames=True,
                                    overlap_strategy="skip")
    assert np.allclose(refined[0].position, [3.54449334, 3.52869383, 3.51225048, 3.38714712, 3.48588412])
    assert np.allclose(refined[1].position, [4.96700319, 4.99771575, 5.04086914, 5.0066495,  4.99092852])

    # gapped data, skip missing frames
    refined = refine_lines_gaussian(gapped_lines, window=3, refine_missing_frames=False,
                                    overlap_strategy="ignore")
    assert np.allclose(refined[0].position, [3.54449334, 3.52869383, 3.38714712, 3.48588412])
    assert np.allclose(refined[1].position, [4.96700319, 4.99771575, 5.0066495,  4.99092852])

    # mixed length lines, no overlap
    refined = refine_lines_gaussian(mixed_lines, window=3, refine_missing_frames=True,
                                    overlap_strategy="skip")
    assert np.allclose(refined[0].position, [3.52869383, 3.51225048])
    assert np.allclose(refined[1].position, [4.96700319, 4.99771575, 5.04086914, 5.0066495, 4.99092852])

    # mixed length lines, track windows overlap
    with pytest.warns(UserWarning):
        refined = refine_lines_gaussian(mixed_lines, window=10, refine_missing_frames=True,
                                        overlap_strategy="skip")
    # all frames in mixed_lines[0] overlap with second track, all skipped
    assert len(refined) == 1
    # 2 frames in mixed_lines[1] overlap with first track, 2 skipped, 3 fitted
    assert np.allclose(refined[0].position, [4.94659924, 5.00920806, 4.97724526])
