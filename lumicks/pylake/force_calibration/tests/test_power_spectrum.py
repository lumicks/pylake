import numpy as np
import pytest

from lumicks.pylake.force_calibration.power_spectrum import PowerSpectrum


def test_power_spectrum_bad_shape():
    with pytest.raises(
        ValueError,
        match=r"Only 1D arrays of data are supported. You provided a 2D array of shape \(10, 2\)",
    ):
        _ = PowerSpectrum(np.ones((10, 2)), 1, window_seconds=1)


def test_power_spectrum_variance():
    """Testing the variance of Welch PSD estimates"""
    sigma, sample_rate = 40, 100

    np.random.seed(90083773)
    ps = PowerSpectrum(sigma * np.random.normal(size=(1000000)), sample_rate, window_seconds=1)
    avg_variance = np.mean(ps._variance[1:-1])  # First and last bin have twice the variance

    # scaling_factor = (2.0 / sample_rate)
    # average power is: sigma**2 * scaling_factor
    # its variance is: (sigma**2 * scaling_factor)**2 (1024 here)
    np.testing.assert_allclose(avg_variance, 1019.8465525703106)

    mask = np.logical_and(ps.frequency > 10, ps.frequency <= 25)
    np.testing.assert_equal(
        ps.in_range(frequency_min=10.0, frequency_max=25.0)._variance, ps._variance[mask]
    )

    mask = np.logical_or(ps.frequency < 10, ps.frequency >= 25)
    np.testing.assert_equal(ps._exclude_range([[10, 25]])._variance, ps._variance[mask])

    # A single window won't give us a variance
    ps = PowerSpectrum(
        sigma * np.random.normal(size=(2 * sample_rate - 1)), sample_rate, window_seconds=1
    )
    assert ps._variance is None
    assert ps.in_range(10, 15)._variance is None

    # Two windows will (though it will likely not be a great estimate)
    ps = PowerSpectrum(
        sigma * np.random.normal(size=(2 * sample_rate)), sample_rate, window_seconds=1
    )
    assert np.all(ps._variance)


@pytest.mark.parametrize(
    "frequency, num_data, sample_rate",
    [
        [200, 50, 2000],
        [400, 50, 2000],
        [400, 50, 4000],
        [400, 50, 4000],
    ],
)
def test_power_spectrum_attrs(frequency, num_data, sample_rate):
    """Testing the attributes of power spectra"""
    data = np.sin(2.0 * np.pi * frequency / sample_rate * np.arange(num_data))
    power_spectrum = PowerSpectrum(data, sample_rate)

    df = sample_rate / num_data
    frequency_axis = np.arange(0, 0.5 * sample_rate + 1, df)
    np.testing.assert_allclose(power_spectrum.frequency, frequency_axis)
    np.testing.assert_allclose(power_spectrum.num_samples(), len(frequency_axis))

    # Functional test
    determined_frequency = power_spectrum.frequency[np.argmax(power_spectrum.power)]
    np.testing.assert_allclose(
        determined_frequency, frequency
    )  # Is the frequency at the right position?

    # Is all the power at this frequency? (Valid when dealing with multiples of the sample rate)
    np.testing.assert_allclose(
        power_spectrum.power[np.argmax(power_spectrum.power)],
        np.sum(power_spectrum.power),
        atol=1e-16,
    )
    np.testing.assert_allclose(power_spectrum.total_duration, num_data / sample_rate)
    np.testing.assert_allclose(power_spectrum.frequency_bin_width, sample_rate / num_data)


@pytest.mark.parametrize(
    "frequency, num_data, sample_rate, num_blocks, f_blocked, p_blocked",
    [
        # fmt: off
        [200, 50, 2000, 4, [100, 340, 580, 820], [1.04166667e-3, 0, 0, 0]],
        [400, 50, 2000, 4, [100, 340, 580, 820], [0, 1.04166667e-3, 0, 0]],
        [400, 50, 4000, 4, [200, 680, 1160, 1640], [5.20833333e-04, 0, 0, 0]],
        [400, 50, 4000, 8, [80, 320, 560, 800, 1040, 1280, 1520, 1760], [0, 1.04166667e-03, 0, 0, 0, 0, 0, 0]],
        [400, 50, 4000, 7, [80, 320, 560, 800, 1040, 1280, 1520, 1760], [0, 1.04166667e-03, 0, 0, 0, 0, 0, 0]],
        # fmt: on
    ],
)
def test_power_spectrum_blocking(
    frequency, num_data, sample_rate, num_blocks, f_blocked, p_blocked
):
    """Functional test whether the results of blocking the power spectrum are correct"""
    data = np.sin(2.0 * np.pi * frequency / sample_rate * np.arange(num_data)) / np.sqrt(2)
    power_spectrum = PowerSpectrum(data, sample_rate)

    downsampling_factor = len(power_spectrum.frequency) // num_blocks
    blocked = power_spectrum.downsampled_by(downsampling_factor)
    np.testing.assert_allclose(blocked.frequency, f_blocked)
    np.testing.assert_allclose(blocked.power, p_blocked, atol=1e-16)
    np.testing.assert_allclose(blocked.num_samples(), len(f_blocked))
    np.testing.assert_allclose(len(power_spectrum.power), num_data // 2 + 1)
    np.testing.assert_allclose(
        blocked.num_points_per_block, np.floor(len(power_spectrum.power) / num_blocks)
    )
    np.testing.assert_allclose(
        blocked.frequency_bin_width, blocked.frequency[1] - blocked.frequency[0]
    )

    # Downsample again and make sure the num_points_per_block is correct
    dual_blocked = blocked.downsampled_by(2)
    np.testing.assert_allclose(dual_blocked.num_points_per_block, blocked.num_points_per_block * 2)
    np.testing.assert_allclose(
        dual_blocked.frequency_bin_width, dual_blocked.frequency[1] - dual_blocked.frequency[0]
    )


@pytest.mark.parametrize(
    "amp, frequency, data_duration, sample_rate, multiple",
    [
        [1.0, 36.33, 5, 78125, 5],
        [1.0, 36.33, 5, 78125, 5],
        [1.0, 37.33, 5, 78125, 5],
        [1.0, 37.33, 10, 78125, 5],
        [2.0, 37.33, 5, 78125, 5],
        [3.0, 37.33, 5, 78125, 5],
        [3.0, 1000.5, 5, 78125, 5],
    ],
)
def test_windowing_sine_wave(amp, frequency, data_duration, sample_rate, multiple):
    """Functional test whether the results of windowing the power spectrum are correct"""
    data = amp * np.sin(
        2.0 * np.pi * frequency / sample_rate * np.arange(data_duration * sample_rate)
    )

    window_duration = multiple / frequency
    power_spectrum_windowed = PowerSpectrum(data, sample_rate, window_seconds=window_duration)

    # Windowing with the correct window size makes the full power end up in a single bin (this is
    # the property we're after).
    delta_freq = power_spectrum_windowed.frequency[1] - power_spectrum_windowed.frequency[0]
    max_idx = np.argmax(power_spectrum_windowed.power)
    power, freq = power_spectrum_windowed.power[max_idx], power_spectrum_windowed.frequency[max_idx]
    np.testing.assert_allclose(freq, frequency, atol=delta_freq)
    np.testing.assert_allclose(power * delta_freq, amp**2 / 2, rtol=1e-4)

    # Check whether we report the correct amount of averaging
    num_points_per_window = int(np.round((window_duration * sample_rate)))
    assert power_spectrum_windowed.num_points_per_block == len(data) // num_points_per_window
    np.testing.assert_allclose(
        power_spectrum_windowed.frequency_bin_width,
        power_spectrum_windowed.frequency[1] - power_spectrum_windowed.frequency[0],
    )

    # When we don't use windowing, we leak when the driving frequency is not an exact multiple of
    # the sampling rate.
    power_spectrum = PowerSpectrum(data, sample_rate)
    low_power = np.max(power_spectrum.power) * (
        power_spectrum.frequency[1] - power_spectrum.frequency[0]
    )
    assert low_power < (power * delta_freq)
    np.testing.assert_allclose(
        power_spectrum_windowed.frequency_bin_width,
        power_spectrum_windowed.frequency[1] - power_spectrum_windowed.frequency[0],
    )


def test_windowing_too_long_duration():
    sample_rate = 78125
    data = np.sin(2.0 * np.pi * 36.33 / sample_rate * np.arange(2 * sample_rate))
    ref_power_spectrum = PowerSpectrum(data, sample_rate)

    with pytest.warns(RuntimeWarning):
        power_spectrum_windowed = PowerSpectrum(data, sample_rate, window_seconds=50000000)

    assert power_spectrum_windowed.num_points_per_block == 1
    np.testing.assert_allclose(ref_power_spectrum.frequency, power_spectrum_windowed.frequency)
    np.testing.assert_allclose(ref_power_spectrum.power, power_spectrum_windowed.power)
    np.testing.assert_allclose(
        power_spectrum_windowed.frequency_bin_width,
        power_spectrum_windowed.frequency[1] - power_spectrum_windowed.frequency[0],
    )


def test_windowing_negative_duration():
    with pytest.raises(ValueError):
        PowerSpectrum([], 78125, window_seconds=-1)


@pytest.mark.parametrize(
    "frequency, num_data, sample_rate, num_blocks, f_min, f_max",
    [
        [200, 50, 2000, 4, 0, 10000],
        [400, 50, 2000, 4, 0, 10000],
        [400, 50, 4000, 4, 0, 10000],
        [400, 50, 4000, 8, 0, 10000],
        [400, 50, 4000, 8, 0, 100],
        [400, 50, 4000, 8, 0, 100],
    ],
)
def test_in_range(frequency, num_data, sample_rate, num_blocks, f_min, f_max):
    data = np.sin(2.0 * np.pi * frequency / sample_rate * np.arange(num_data))
    power_spectrum = PowerSpectrum(data, sample_rate)

    power_subset = power_spectrum.in_range(f_min, f_max)
    assert id(power_subset) != id(power_spectrum)
    assert id(power_subset.frequency) != id(power_spectrum.frequency)
    assert id(power_subset.power) != id(power_spectrum.power)

    # Note that this different from the slice behaviour you'd normally expect
    mask = (power_spectrum.frequency > f_min) & (power_spectrum.frequency <= f_max)
    np.testing.assert_allclose(power_subset.frequency, power_spectrum.frequency[mask])
    np.testing.assert_allclose(power_subset.power, power_spectrum.power[mask], atol=1e-16)
    np.testing.assert_allclose(power_spectrum.total_duration, power_subset.total_duration)
    np.testing.assert_allclose(power_spectrum.sample_rate, power_subset.sample_rate)


def test_plot():
    ps = PowerSpectrum(np.sin(2.0 * np.pi * 100 / 78125 * np.arange(100)), 78125)
    ps.plot()
    ps = ps.downsampled_by(2)
    ps.plot()


def test_replace_spectrum():
    power_spectrum = PowerSpectrum(np.arange(10), 5)
    replaced = power_spectrum.with_spectrum(np.arange(6))

    np.testing.assert_allclose(power_spectrum.frequency, replaced.frequency)
    np.testing.assert_allclose(replaced.power, np.arange(6), atol=1e-16)
    np.testing.assert_allclose(replaced.num_points_per_block, 1)
    np.testing.assert_allclose(replaced.sample_rate, power_spectrum.sample_rate)
    np.testing.assert_allclose(replaced.total_duration, power_spectrum.total_duration)

    with pytest.raises(ValueError, match="New power spectral density vector has incorrect length"):
        power_spectrum.with_spectrum(np.arange(7))


@pytest.mark.parametrize(
    "exclusion_ranges, result_frequency, result_power",
    [
        [[], [1, 3, 5, 7, 9, 11], [10, 30, 50, 70, 90, 110]],  # Empty exclusion range
        [[(3, 5)], [1, 5, 7, 9, 11], [10, 50, 70, 90, 110]],
        [[(3, 6)], [1, 7, 9, 11], [10, 70, 90, 110]],
        [[(3, 6), (3, 6)], [1, 7, 9, 11], [10, 70, 90, 110]],  # Same exclusion range twice
        [[(5, 6), (3, 8)], [1, 9, 11], [10, 90, 110]],  # Overlapping range
        [[(3, 8), (5, 6)], [1, 9, 11], [10, 90, 110]],  # Overlapping range (different order)
        [[(3, 6), (9, 10)], [1, 7, 11], [10, 70, 110]],  # Disjoint range
    ],
)
def test_exclusions(exclusion_ranges, result_frequency, result_power):
    ps = PowerSpectrum([1], 78125, unit="V")
    ps.frequency = np.arange(1, 12, 2)
    ps.power = np.arange(10, 120, 20)
    ps._variance = None
    excluded_range = ps._exclude_range(exclusion_ranges)
    np.testing.assert_allclose(excluded_range.frequency, result_frequency)
    np.testing.assert_allclose(excluded_range.power, result_power)


def test_identify_peaks_args():
    power_spectrum = PowerSpectrum([1], 5)
    power_spectrum.frequency = np.arange(10)
    power_spectrum.power = np.ones_like(power_spectrum.frequency)
    with pytest.raises(ValueError, match="baseline cannot be negative"):
        power_spectrum.identify_peaks(lambda f: np.ones_like(f), baseline=-1.0)

    with pytest.raises(ValueError, match="peak_cutoff must be greater than baseline value"):
        power_spectrum.identify_peaks(lambda f: np.ones_like(f), baseline=10.0, peak_cutoff=9.0)

    with pytest.raises(
        ValueError,
        match="identify_peaks only works if the power spectrum is not blocked / averaged",
    ):
        power_spectrum.downsampled_by(2).identify_peaks(lambda f: np.ones_like(f))


@pytest.mark.parametrize(
    "power, result_frequency",
    [
        [np.ones(10), []],  # Nothing above peak_cutoff
        [[0, 1, 100, 1, 0, 0, 0, 0, 0, 0], [(1, 4)]],  # Single peak
        [[0, 0, 100, 0, 0, 0, 0, 0, 0, 0], [(2, 3)]],  # Single peak
        [[1, 100, 1, 0, 0, 0, 0, 0, 0, 0], [(0, 3)]],  # Single peak, near left edge
        [[100, 0, 1, 0, 0, 0, 0, 0, 0, 0], [(0, 1)]],  # Single peak, touching left edge
        [[1, 0, 1, 0, 0, 0, 0, 0, 100, 1], [(8, 10)]],  # Single peak, near right edge
        [[1, 0, 1, 0, 0, 0, 0, 0, 1, 100], [(8, 10)]],  # Single peak, touching right edge
        [[1, 0, 1, 0, 0, 0, 0, 0, 0, 100], [(9, 10)]],  # Single peak, touching right edge
        [[100, 100, 1, 0, 0, 0, 0, 0, 1, 0], [(0, 3)]],  # Large peak, at edge
        [[0, 1, 100, 1, 100, 1, 0, 0, 1, 0], [(1, 6)]],  # Double peak, concatenated
        [[0, 1, 100, 1, 0, 0, 0, 0, 1, 100], [(1, 4), (8, 10)]],  # Double peak
        [[0, 1, 100, 0, 0, 0, 0, 100, 1, 0], [(1, 3), (7, 9)]],  # Double peak
        [[2, 2, 100, 2, 2, 2, 2, 2, 2, 100], [(0, 10)]],  # Nothing below baseline
    ],
)
def test_identify_peaks(power, result_frequency):
    power_spectrum = PowerSpectrum([1], 5)
    power_spectrum.frequency = np.arange(10)
    power_spectrum.power = power

    assert power_spectrum.identify_peaks(lambda f: np.ones_like(f)) == result_frequency
