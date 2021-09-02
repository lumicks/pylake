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
        [[[0.5, 36.9], [1, 41.0]], 1.0, 41.0, 36, 6, 2],  # Does it pick up on the second peak?
        [[[0.5, 36.9], [-1, 41.0]], 1.0, 41.0, 36, 6, 2],  # Does it pick up on the second peak?
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


def test_frequency_validation():
    """We want to validate that the driving peak was actually in the search range. If not, that
    means something may have gone critically wrong. It's better to fail early than to silently
    calculate a calibration that has a relatively large area because the fitting range
    did not encompass the peak."""
    sines = [np.sin(f * 2.0 * np.pi * np.arange(0, 1, 1.0 / 78125)) for f in [32, 70]]
    driving_data = np.sum(np.vstack(sines), axis=0)
    with pytest.raises(
        RuntimeError, match="Did not manage to find driving peak in spectral search range"
    ):
        f_drive_guess = 50
        estimate_driving_input_parameters(
            78125, driving_data, f_drive_guess, window_factor=10, f_search=5, n_fit=1
        )

    with pytest.raises(RuntimeError, match="Peak is outside frequency search range"):
        f_drive_guess = 37.1
        estimate_driving_input_parameters(
            78125, driving_data, f_drive_guess, window_factor=10, f_search=5, n_fit=1
        )

    with pytest.raises(RuntimeError, match="Peak is outside frequency search range"):
        f_drive_guess = 26.9
        estimate_driving_input_parameters(
            78125, driving_data, f_drive_guess, window_factor=10, f_search=5, n_fit=1
        )
