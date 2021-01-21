from lumicks.pylake.force_calibration.power_spectrum_calibration import PowerSpectrum, block_average, block_average_std
import numpy as np
import pytest


@pytest.mark.parametrize(
    "data,num_blocks,avg,std",
    [
        [np.arange(10), 5, [0.5, 2.5, 4.5, 6.5, 8.5], [0.5, 0.5, 0.5, 0.5, 0.5]],
        [np.arange(10), 4, [0.5, 2.5, 4.5, 6.5], [0.5, 0.5, 0.5, 0.5]],
        [np.arange(0, 10, 2), 4, [0.0, 2.0, 4.0, 6.0], [0.0, 0.0, 0.0, 0.0]],
        [np.arange(0, 30, 3), 7, [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    ],
)
def test_blocking_separate(data, num_blocks, avg, std):
    assert np.allclose(avg, block_average(data, num_blocks))
    assert np.allclose(std, block_average_std(data, num_blocks))


def test_power_spectrum_none():
    assert PowerSpectrum().f is None
    assert PowerSpectrum().P is None
    assert PowerSpectrum().sampling_rate is None
    assert PowerSpectrum().T_measure is None


@pytest.mark.parametrize(
    "frequency, num_data, sampling_rate",
    [
        [200, 50, 2000],
        [400, 50, 2000],
        [400, 50, 4000],
        [400, 50, 4000],
    ],
)
def test_power_spectrum_attrs(frequency, num_data, sampling_rate):
    """Testing the attributes of power spectra"""
    data = np.sin(2.0 * np.pi * frequency / sampling_rate * np.arange(num_data))
    power_spectrum = PowerSpectrum(data, sampling_rate)

    df = sampling_rate / num_data
    frequency_axis = np.arange(0, .5 * sampling_rate + 1, df)
    assert np.allclose(power_spectrum.f, frequency_axis)
    assert np.allclose(power_spectrum.n_samples(), len(frequency_axis))

    # Functional test
    determined_frequency = power_spectrum.f[np.argmax(power_spectrum.P)]
    assert np.allclose(determined_frequency, frequency)  # Is the frequency at the right position?

    # Is all the power at this frequency? (Only valid when dealing with multiples of the sample rate)
    assert np.allclose(power_spectrum.P[np.argmax(power_spectrum.P)], np.sum(power_spectrum.P))
    assert np.allclose(power_spectrum.T_measure, num_data / sampling_rate)

    # Check serialized properties
    assert np.allclose(power_spectrum.as_dict()["f"], power_spectrum.f)
    assert np.allclose(power_spectrum.as_dict()["P"], power_spectrum.P)


@pytest.mark.parametrize(
    "frequency, num_data, sampling_rate, num_blocks, f_blocked, p_blocked",
    [
        [200, 50, 2000, 4, [100, 340, 580, 820], [1.04166667e-3, 0, 0, 0]],
        [400, 50, 2000, 4, [100, 340, 580, 820], [0, 1.04166667e-3, 0, 0]],
        [400, 50, 4000, 4, [200, 680, 1160, 1640], [5.20833333e-04, 0, 0, 0]],
        [400, 50, 4000, 8, [80, 320, 560, 800, 1040, 1280, 1520, 1760], [0, 1.04166667e-03, 0, 0, 0, 0, 0, 0]],
    ],
)
def test_power_spectrum_blocking(frequency, num_data, sampling_rate, num_blocks, f_blocked, p_blocked):
    """Functional test whether the results of blocking the power spectrum are correct"""
    data = np.sin(2.0 * np.pi * frequency / sampling_rate * np.arange(num_data))
    power_spectrum = PowerSpectrum(data, sampling_rate)

    blocked = power_spectrum.block_averaged(num_blocks)
    assert np.allclose(blocked.f, f_blocked)
    assert np.allclose(blocked.P, p_blocked)
    assert np.allclose(blocked.n_samples(), len(f_blocked))


@pytest.mark.parametrize(
    "frequency, num_data, sampling_rate, num_blocks, f_min, f_max",
    [
        [200, 50, 2000, 4, 0, 10000],
        [400, 50, 2000, 4, 0, 10000],
        [400, 50, 4000, 4, 0, 10000],
        [400, 50, 4000, 8, 0, 10000],
        [400, 50, 4000, 8, 0, 100],
        [400, 50, 4000, 8, 0, 100],
    ],
)
def test_in_range(frequency, num_data, sampling_rate, num_blocks, f_min, f_max):
    data = np.sin(2.0 * np.pi * frequency / sampling_rate * np.arange(num_data))
    power_spectrum = PowerSpectrum(data, sampling_rate)

    power_subset = power_spectrum.in_range(f_min, f_max)
    assert id(power_subset) != id(power_spectrum)
    assert id(power_subset.f) != id(power_spectrum.f)
    assert id(power_subset.P) != id(power_spectrum.P)

    # Note that this different from the slice behaviour you'd normally expect
    mask = (power_spectrum.f > f_min) & (power_spectrum.f <= f_max)
    assert np.allclose(power_subset.f, power_spectrum.f[mask])
    assert np.allclose(power_subset.P, power_spectrum.P[mask])
    assert np.allclose(power_spectrum.T_measure, power_subset.T_measure)
    assert np.allclose(power_spectrum.sampling_rate, power_subset.sampling_rate)
