import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from lumicks.pylake.detail.utilities import downsample


class PowerSpectrum:
    """Power spectrum data for a time series.

    Attributes
    ----------
    frequency : numpy.ndarray
        Frequency values for the power spectrum. [Hz]
    power : numpy.ndarray
        Power values for the power spectrum (typically in V^2/s).
    sample_rate : float
        The sampling rate for the original data. [Hz]
    total_duration : float
        The total duration of the original data. [seconds]
    """

    def __init__(self, data, sample_rate, unit="V"):
        """Power spectrum

        Parameters
        ----------
        data : numpy.ndarray
            Data from which to calculate a power spectrum.
        sample_rate : float
            Sampling rate at which this data was acquired.
        unit : str
            Units the data is in (default: V).
        """
        self.unit = unit
        data = data - np.mean(data)

        # Calculate power spectrum.
        fft = np.fft.rfft(data)
        self.frequency = np.fft.rfftfreq(data.size, 1.0 / sample_rate)
        self.power = (1.0 / sample_rate) * np.square(np.abs(fft)) / data.size

        # Store metadata
        self.sample_rate = sample_rate
        self.total_duration = data.size / sample_rate
        self.num_points_per_block = 1

    def downsampled_by(self, factor, reduce=np.mean):
        """Returns a spectrum downsampled by a given factor."""
        ba = copy(self)
        ba.frequency = downsample(self.frequency, factor, reduce)
        ba.power = downsample(self.power, factor, reduce)
        ba.num_points_per_block = self.num_points_per_block * factor

        return ba

    def in_range(self, frequency_min, frequency_max):
        """Returns part of the power spectrum within a given frequency range."""
        ir = copy(self)
        mask = (self.frequency > frequency_min) & (self.frequency <= frequency_max)
        ir.frequency = self.frequency[mask]
        ir.power = self.power[mask]
        return ir

    def num_samples(self):
        return self.frequency.size

    def with_spectrum(self, power, num_points_per_block=1):
        """Return a copy with a different spectrum"""
        assert len(power) == len(self.power), "Power has incorrect length"

        ps = copy(self)
        ps.power = power
        ps.num_points_per_block = num_points_per_block

        return ps

    def plot(self):
        """Plot power spectrum"""
        plt.plot(self.frequency, self.power)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(f"Power [${self.unit}^2/Hz$]")
        plt.xscale("log")
        plt.yscale("log")
        if self.num_points_per_block:
            plt.title(f"Blocked Power spectrum (N={self.num_points_per_block})")
        else:
            plt.title("Power spectrum")
