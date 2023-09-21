import numpy as np
import pytest

from ..data.mock_confocal import generate_kymo_with_ref

start = np.int64(20e9)
dt = np.int64(62.5e6)


@pytest.fixture(scope="module")
def test_kymo():
    # RGB Kymo with infowave as expected from BL
    image = np.random.poisson(5, size=(5, 10, 3))

    kymo, ref = generate_kymo_with_ref(
        "tester",
        image,
        pixel_size_nm=100,
        start=start,
        dt=dt,
        samples_per_pixel=4,
        line_padding=50,
    )

    return kymo, ref


@pytest.fixture(scope="module")
def truncated_kymo():
    image = np.random.poisson(5, size=(5, 4, 3))

    kymo, ref = generate_kymo_with_ref(
        "truncated",
        image,
        pixel_size_nm=100,
        start=start,
        dt=dt,
        samples_per_pixel=4,
        line_padding=50,
    )
    kymo.start = start - 62500000
    return kymo, ref


@pytest.fixture(scope="module")
def downsampling_kymo():
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8,
    )

    kymo, ref = generate_kymo_with_ref(
        "downsampler",
        image,
        pixel_size_nm=100,
        start=1592916040906356300,
        dt=int(1e9),
        samples_per_pixel=5,
        line_padding=2,
    )

    return kymo, ref


@pytest.fixture(scope="module")
def downsampled_results():
    time_factor = 2
    position_factor = 2
    time_image = np.array(
        [
            [12, 12, 6],
            [0, 0, 6],
            [12, 0, 18],
            [12, 24, 6],
            [12, 24, 6],
        ]
    )
    position_image = np.array(
        [
            [0, 12, 0, 12, 0, 12, 0],
            [12, 12, 12, 12, 12, 12, 0],
        ]
    )
    both_image = np.array([[12, 12, 12], [24, 24, 24]])

    return time_factor, position_factor, time_image, position_image, both_image
