from lumicks.pylake.force_calibration.detail.driving_input import estimate_driving_input_parameters
import numpy as np
import pytest


@pytest.mark.parametrize(
    "sine_waves, expected_amp, expected_freq, guess, f_search, n_fit",
    [
        [[[0.5, 36.9], [1, 50]], 0.5, 36.9, 36, 5, 2],
        [[[0.34, 36.7], [1, 50]], 0.34, 36.7, 36, 5, 2],
        [[[0.45, 36.554], [1, 50]], 0.45, 36.554, 36, 5, 2],
        [[[0.5, 36.9], [1, 42.0]], 0.5, 36.9, 36, 5, 2],  # Does window exclude the bigger peak?
        [[[0.5, 36.9], [1, 42.0]], 1.0, 42.0, 36, 10, 2],  # Bigger window includes second peak
        [[[0.5, 36.9], [1, 41.0]], 1.0, 41.0, 36, 5, 2],  # Does it pick up on the second peak?
        [[[0.5, 36.9], [-1, 41.0]], 1.0, 41.0, 36, 5, 2],  # Does it pick up on the second peak?
        [[[0.5, 36.9], [1, 42.0]], 0.617576, 41.999864, 36, 10, 10],  # Fit range has both peaks
        [[[0.5, 2.4], [0, 50]], 0.5, 2.4, 3, 5, 2],  # Does it reject offset correctly?
    ],
)
def test_driving_input_estimation(sine_waves, expected_amp, expected_freq, guess, f_search, n_fit):
    sample_rate = 78125
    duration = 10
    t = np.arange(0, duration, 1.0 / sample_rate)
    y = np.ones(t.shape)

    for amp, freq in sine_waves:
        y += amp * np.cos(2 * np.pi * freq * t)

    amp, freq = estimate_driving_input_parameters(
        sample_rate,
        y,
        guess,
        window_factor=10,
        f_search=f_search,
        n_fit=n_fit,
    )

    np.testing.assert_allclose(amp, expected_amp, rtol=1e-6)
    np.testing.assert_allclose(freq, expected_freq, rtol=1e-6)
