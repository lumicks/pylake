import numpy as np
import pytest

from lumicks.pylake.kymo import _kymo_from_image_stack
from lumicks.pylake.image_stack import ImageStack
from lumicks.pylake.detail.widefield import Tether, TiffStack
from lumicks.pylake.tests.data.mock_widefield import MockTiffFile


def make_frame_times(n_frames, rate=10, step=8, start=10, framerate_jitter=0, step_jitter=0):
    def noise(jitter, j):
        if j / rate % 2:
            return 0
        else:
            return jitter

    return [
        [
            f"{j + noise(framerate_jitter, j)}",
            f"{(j + 1) + noise(framerate_jitter, (j + 1))}",
            step + noise(step_jitter, j),
        ]
        for j in range(start, start + n_frames * rate, rate)
    ]


def create_mock_stack(
    n_frames=1, shape=None, image=None, framerate_jitter=0, step_jitter=0, tether_ends=None
):
    image = np.ones(shape) if image is None else image
    times = make_frame_times(n_frames, framerate_jitter=framerate_jitter, step_jitter=step_jitter)
    tether = tether_ends if tether_ends is None else Tether((0, 0), tether_ends)
    fake_tiff = TiffStack(
        [MockTiffFile(data=[image] * n_frames, times=times)],
        align_requested=False,
        tether=tether,
    )
    return ImageStack.from_dataset(fake_tiff)


def gaussian_2d(shape=(15, 15), limit=(3.0, 3.0), mean=(0.0, 0.0), std=(1.0, 1.0), rho=0.0):
    """Create a 2D gaussian with two different standard deviations

    Parameters
    ----------
    shape : tuple of int
        The shape of the resulting 2D array for x and y.
    limit : tuple of float
        The absolute values of standard deviation up to where the values for the gaussians in x and
        y are calculated.
    mean : tuple of float
        The mean of the guassians in values of standard deviation in x and y. 0.0 corresponds to the
        image center of the corresponding axis.
    std : tuple of float
        The standard deviation of the two gaussians in x and y.
    rho : float
        The correlation of the two gaussians. Accepted values are in the open interval ( -1 , 1 ).
    """
    from scipy.stats import multivariate_normal

    sx, sy = std
    cov = np.array([[sx**2, sx * sy * rho], [sx * sy * rho, sy**2]])
    mesh = np.dstack(np.mgrid[[slice(-dx, dx, n * 1j) for (dx, n) in zip(limit, shape)]])
    return multivariate_normal(mean=mean, cov=cov).pdf(mesh)


# Test conditional errors for insufficient image stacks as input
def test_error_frame_rate_not_constant():
    stack = create_mock_stack(3, (3, 3, 3), framerate_jitter=1)
    with pytest.raises(
        ValueError, match="The frame rate of the images of the image stack is not constant."
    ):
        _kymo_from_image_stack(stack)


@pytest.mark.filterwarnings("ignore:This image stack contains a non-constant exposure time")
def test_error_exposure_time_not_constant():
    stack = create_mock_stack(3, (3, 3, 3), step_jitter=1)
    with pytest.raises(
        ValueError, match="The exposure time of the images of the image stack is not constant."
    ):
        _kymo_from_image_stack(stack)


def test_error_tether_not_exists():
    stack = create_mock_stack(3, (3, 3, 3))
    with pytest.raises(ValueError, match="The image stack does not have a tether."):
        _kymo_from_image_stack(stack)


def test_error_negative_linewidth():
    stack = create_mock_stack(3, (3, 3, 3), tether_ends=((0, 0), (4, 0)))
    with pytest.raises(
        ValueError, match="The requested number of `adjacent_lines` must not be negative."
    ):
        _kymo_from_image_stack(stack, adjacent_lines=-1)


def test_error_tether_linewidth_exceeds_image():
    stack = create_mock_stack(3, (3, 3, 3), tether_ends=((0, 0), (4, 0)))
    with pytest.raises(
        ValueError,
        match="The number of `adjacent_lines` exceed the size of the image stack images.",
    ):
        _kymo_from_image_stack(stack, adjacent_lines=3)

    stack = create_mock_stack(3, (3, 3, 3), tether_ends=((0, 2), (4, 2)))
    with pytest.raises(
        ValueError,
        match="The number of `adjacent_lines` exceed the size of the image stack images.",
    ):
        _kymo_from_image_stack(stack, adjacent_lines=3)


# Test proper shape of kymo
def test_shape_of_kymo():
    # 5 pixels, 4 frames, 3 channels
    stack = create_mock_stack(4, (5, 5, 3), tether_ends=((0, 2), (4, 2)))
    kymo = _kymo_from_image_stack(stack)
    assert kymo.get_image().shape == (5, 4, 3)

    # 5 pixels (rounded), 4 frames, 3 channels
    stack = create_mock_stack(4, (5, 5, 3), tether_ends=((0.5, 2), (4.5, 2)))
    kymo = _kymo_from_image_stack(stack)
    assert kymo.get_image().shape == (5, 4, 3)

    # 3 pixels, 4 frames, 3 channels
    stack = create_mock_stack(4, (5, 5, 3), tether_ends=((1, 2), (3, 2)))
    kymo = _kymo_from_image_stack(stack)
    assert kymo.get_image().shape == (3, 4, 3)

    # 3 pixels, 10 frames, 1 channel
    stack = create_mock_stack(10, (5, 5), tether_ends=((1, 2), (3, 2)))
    kymo = _kymo_from_image_stack(stack)
    assert kymo.get_image().shape == (3, 10, 3)


@pytest.mark.parametrize("adjacent_lines", [0, 1, 2])
def test_data_identity_horizonal_tether(adjacent_lines, reduce=np.mean):
    x = 15
    y = 25
    # make image with two gaussians
    image = np.swapaxes(
        np.array([gaussian_2d(shape=(x, y), mean=(-1.0, -1.5), std=(1, 0.8))] * 3), 0, 2
    )
    image += np.swapaxes(
        np.array([gaussian_2d(shape=(x, y), mean=(1.0, 1.0), std=(0.75, 1))] * 3), 0, 2
    )
    ymin = y // 2 - adjacent_lines
    ymax = y // 2 + adjacent_lines + 1
    imageline = reduce(image[ymin:ymax], axis=0)
    stack = create_mock_stack(5, image=image, tether_ends=((0, y // 2), (x - 1, y // 2)))
    kymo = _kymo_from_image_stack(stack, adjacent_lines=adjacent_lines, reduce=reduce)
    for frameline in np.swapaxes(kymo.get_image(), 0, 1):
        assert np.array_equal(frameline, imageline)
