import pytest
import numpy as np
from lumicks.pylake.kymotracker.kymotracker import track_greedy
from lumicks.pylake.tests.data.mock_confocal import generate_kymo


def generate_gaussian_line(vel, dt, dx, sigma=0.3, area=10, samples_per_pixel=10):
    """Generate a Gaussian test line with a certain velocity"""
    time = np.arange(0, 2, dt)
    time_mesh, coord_mesh = np.meshgrid(time, np.arange(0, 4, dx))
    img_data = (area * np.exp(-(((vel * time_mesh - coord_mesh + 1) / sigma) ** 2))).astype(int)
    ns_per_sample = int(1e9 * dt / len(time) / samples_per_pixel)
    return generate_kymo(
        "chan",
        img_data,
        dt=ns_per_sample,
        samples_per_pixel=samples_per_pixel,
        line_padding=0,
        pixel_size_nm=1e3 * dx,
    )


@pytest.mark.parametrize(
    "vel, dt, dx",
    [
        [1.0, 0.1, 0.2],
        [0.5, 0.1, 0.2],
    ],
)
def test_kymotracker_positional_scaling(vel, dt, dx):
    """Integration test to make sure position and velocity parameters are correctly scaled"""
    kymo = generate_gaussian_line(vel=vel, dt=dt, dx=dx)
    traces = track_greedy(kymo, "red", pixel_threshold=3, line_width=1, sigma=0.01, vel=vel)
    ref_seconds = np.arange(0.0, 2.0, dt)
    ref_positions = 1 + vel * ref_seconds
    np.testing.assert_allclose(traces[0].seconds, ref_seconds)
    np.testing.assert_allclose(traces[0].position, ref_positions, rtol=1e-2)

    # Check whether a wrong velocity also fails to track the line
    traces = track_greedy(kymo, "red", pixel_threshold=3, line_width=1, sigma=0.01, vel=2 * vel)
    np.testing.assert_equal(len(traces[0].seconds), 1)
    np.testing.assert_equal(len(traces[0].position), 1)

    # When sigma is large, we expect the line to be strung together despite the velocity being zero
    traces = track_greedy(kymo, "red", pixel_threshold=3, line_width=1, sigma=0.5 * vel * dx, vel=0)
    np.testing.assert_allclose(traces[0].seconds, ref_seconds)
    np.testing.assert_allclose(traces[0].position, ref_positions, rtol=1e-2)
