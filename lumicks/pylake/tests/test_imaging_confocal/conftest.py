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
