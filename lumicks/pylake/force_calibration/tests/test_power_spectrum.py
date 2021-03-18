from lumicks.pylake.force_calibration.power_spectrum import PowerSpectrum
from matplotlib.testing.decorators import cleanup
import numpy as np
import pytest


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
    assert np.allclose(power_spectrum.frequency, frequency_axis)
    assert np.allclose(power_spectrum.num_samples(), len(frequency_axis))

    # Functional test
    determined_frequency = power_spectrum.frequency[np.argmax(power_spectrum.power)]
    assert np.allclose(determined_frequency, frequency)  # Is the frequency at the right position?

    # Is all the power at this frequency? (Valid when dealing with multiples of the sample rate)
    assert np.allclose(power_spectrum.power[np.argmax(power_spectrum.power)], np.sum(power_spectrum.power))
    assert np.allclose(power_spectrum.total_duration, num_data / sample_rate)


@pytest.mark.parametrize(
    "frequency, num_data, sample_rate, num_blocks, f_blocked, p_blocked",
    [
        [200, 50, 2000, 4, [100, 340, 580, 820], [1.04166667e-3, 0, 0, 0]],
        [400, 50, 2000, 4, [100, 340, 580, 820], [0, 1.04166667e-3, 0, 0]],
        [400, 50, 4000, 4, [200, 680, 1160, 1640], [5.20833333e-04, 0, 0, 0]],
        [400, 50, 4000, 8, [80, 320, 560, 800, 1040, 1280, 1520, 1760], [0, 1.04166667e-03, 0, 0, 0, 0, 0, 0]],
        [400, 50, 4000, 7, [80, 320, 560, 800, 1040, 1280, 1520, 1760], [0, 1.04166667e-03, 0, 0, 0, 0, 0, 0]],
    ],
)
def test_power_spectrum_blocking(frequency, num_data, sample_rate, num_blocks, f_blocked, p_blocked):
    """Functional test whether the results of blocking the power spectrum are correct"""
    data = np.sin(2.0 * np.pi * frequency / sample_rate * np.arange(num_data))
    power_spectrum = PowerSpectrum(data, sample_rate)

    downsampling_factor = len(power_spectrum.frequency) // num_blocks
    blocked = power_spectrum.downsampled_by(downsampling_factor)
    assert np.allclose(blocked.frequency, f_blocked)
    assert np.allclose(blocked.power, p_blocked)
    assert np.allclose(blocked.num_samples(), len(f_blocked))
    assert np.allclose(len(power_spectrum.power), num_data // 2 + 1)
    assert np.allclose(blocked.num_points_per_block, np.floor(len(power_spectrum.power) / num_blocks))

    # Downsample again and make sure the num_points_per_block is correct
    dual_blocked = blocked.downsampled_by(2)
    assert np.allclose(dual_blocked.num_points_per_block, blocked.num_points_per_block * 2)


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
    assert np.allclose(power_subset.frequency, power_spectrum.frequency[mask])
    assert np.allclose(power_subset.power, power_spectrum.power[mask])
    assert np.allclose(power_spectrum.total_duration, power_subset.total_duration)
    assert np.allclose(power_spectrum.sample_rate, power_subset.sample_rate)


@cleanup
def test_plot():
    ps = PowerSpectrum(np.sin(2.0 * np.pi * 100 / 78125 * np.arange(100)), 78125)
    ps.plot()


@cleanup
def test_replace_spectrum():
    power_spectrum = PowerSpectrum(np.arange(10), 5)
    replaced = power_spectrum.with_spectrum(np.arange(6))

    assert np.allclose(power_spectrum.frequency, replaced.frequency)
    assert np.allclose(replaced.power, np.arange(6))
    assert np.allclose(replaced.num_points_per_block, 1)
    assert np.allclose(replaced.sample_rate, power_spectrum.sample_rate)
    assert np.allclose(replaced.total_duration, power_spectrum.total_duration)
