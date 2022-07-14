import pytest
import numpy as np
from lumicks.pylake.correlated_stack import CorrelatedStack
from lumicks.pylake.detail.widefield import TiffStack, Tether
from lumicks.pylake.tests.data.mock_widefield import MockTiffFile
from lumicks.pylake.kymo import _kymo_from_correlated_stack


def make_frame_times(n_frames, rate=10, step=8, start=10, framerate_jitter=0, step_jitter=0):
    def noise(jitter, j):
        if j / rate % 2:
            return 0
        else:
            return jitter

    return [
        [
            f"{j + noise(framerate_jitter, j)}",
            f"{j + step + noise(framerate_jitter, j) + noise(step_jitter, j)}",
        ]
        for j in range(start, start + n_frames * rate, rate)
    ]


def create_mock_corrstack(n_frames, shape, framerate_jitter=0, step_jitter=0, tether_ends=None):
    times = make_frame_times(n_frames, framerate_jitter=framerate_jitter, step_jitter=step_jitter)
    tether = tether_ends if tether_ends is None else Tether((0, 0), tether_ends)
    fake_tiff = TiffStack(
        [MockTiffFile(data=[np.ones(shape)] * n_frames, times=times)],
        align_requested=False,
        tether=tether,
    )
    return CorrelatedStack.from_dataset(fake_tiff)


# Test conditional errors for unsufficient correlated stacks as input
corrstack = create_mock_corrstack(3, (3, 3, 3), framerate_jitter=1)
with pytest.raises(
    ValueError, match="The frame rate of the images of the correlated stack is not constant."
):
    _kymo_from_correlated_stack(corrstack)

corrstack = create_mock_corrstack(3, (3, 3, 3), step_jitter=1)
with pytest.raises(
    ValueError, match="The exposure time of the images of the correlated stack is not constant."
):
    _kymo_from_correlated_stack(corrstack)

corrstack = create_mock_corrstack(3, (3, 3, 3))
with pytest.raises(ValueError, match="The correlated stack does not have a tether."):
    _kymo_from_correlated_stack(corrstack)

corrstack = create_mock_corrstack(3, (3, 3, 3), tether_ends=((0, 1), (4, 1)))
with pytest.raises(
    ValueError,
    match="The requested `width` of the line exceeds the size of the correlated stack images.",
):
    _kymo_from_correlated_stack(corrstack, width=5)

# Test proper creation of kymo
corrstack = create_mock_corrstack(4, (5, 5, 3), tether_ends=((0, 1), (4, 1)))
kymo = _kymo_from_correlated_stack(corrstack)
assert kymo.get_image().shape == (5, 4, 3)


corrstack = create_mock_corrstack(4, (5, 5, 3), tether_ends=((1, 1), (3, 1)))
kymo = _kymo_from_correlated_stack(corrstack)
assert kymo.get_image().shape == (3, 4, 3)


corrstack = create_mock_corrstack(10, (5, 5, 3), tether_ends=((1, 1), (3, 1)))
kymo = _kymo_from_correlated_stack(corrstack)
assert kymo.get_image().shape == (3, 10, 3)


corrstack = create_mock_corrstack(4, (5, 5, 3), tether_ends=((1, 2), (3, 2)))


# Test if width (and reduce) is respected and maps like 0 -> 1, 1 -> 1, 2 -> 3
@pytest.mark.parametrize("width,pixels", [(0, 1), (1, 1), (2, 3), (3, 3), (4, 5)])
def test_width(width, pixels):
    kymo = _kymo_from_correlated_stack(corrstack, width=width, reduce=np.sum)
    np.all(kymo.get_image() == pixels)
