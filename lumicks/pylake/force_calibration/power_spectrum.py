import warnings
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
        Power values for the power spectrum (typically in V^2/Hz).
    sample_rate : float
        The sampling rate for the original data. [Hz]
    total_duration : float
        The total duration of the original data. [seconds]
    """

    def __init__(self, data, sample_rate, unit="V", window_seconds=None):
        """Power spectrum

        Parameters
        ----------
        data : numpy.ndarray
            Data from which to calculate a power spectrum.
        sample_rate : float
            Sampling rate at which this data was acquired [Hz].
        unit : str
            Units the data is in (default: V).
        window_seconds : float
            Window duration [seconds]. When specified the data is divided into blocks of length
            window_seconds. Power spectra are computed for each block after which they are
            averaged. If omitted, no windowing is used.
        """
        if window_seconds is not None and window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        def squared_fft(d):
            return np.square(np.abs(np.fft.rfft(d)))

        self.unit = unit
        data = data - np.mean(data)

        # Calculate power spectrum for slices of data.
        num_points_per_window = (
            int(np.round(window_seconds * sample_rate)) if window_seconds else len(data)
        )
        if num_points_per_window > len(data):
            warnings.warn(RuntimeWarning("Longer window than data duration: not using windowing."))
            num_points_per_window = len(data)

        squared_fft_chunks = [
            squared_fft(
                data[chunk_idx * num_points_per_window : (chunk_idx + 1) * num_points_per_window]
            )
            for chunk_idx in np.arange(len(data) // num_points_per_window)
        ]
        squared_fft = np.mean(squared_fft_chunks, axis=0)

        self.frequency = np.fft.rfftfreq(num_points_per_window, 1.0 / sample_rate)
        self.power = (2.0 / sample_rate) * squared_fft / num_points_per_window

        # Store metadata
        self.sample_rate = sample_rate
        self.total_duration = data.size / sample_rate
        self.num_points_per_block = len(squared_fft_chunks)

    def downsampled_by(self, factor, reduce=np.mean):
        """Returns a spectrum downsampled by a given factor."""
        ba = copy(self)
        ba.frequency = downsample(self.frequency, factor, reduce)
        ba.power = downsample(self.power, factor, reduce)
        ba.num_points_per_block = self.num_points_per_block * factor

        return ba

    def _exclude_range(self, excluded_ranges):
        """Exclude given frequency ranges from the power spectrum.

        This function can be used to exclude noise peaks.

        Parameters
        ----------
        excluded_ranges : list of tuple of float
            List of ranges to exclude specified as a list of (frequency_min, frequency_max)."""
        if not excluded_ranges:
            return copy(self)

        ps = copy(self)
        indices = np.logical_and.reduce(
            [(ps.frequency < f_min) | (ps.frequency >= f_max) for f_min, f_max in excluded_ranges]
        )

        ps.frequency = ps.frequency[indices]
        ps.power = ps.power[indices]
        return ps

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

    def plot(self, **kwargs):
        """Plot power spectrum

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`.
        """
        plt.plot(self.frequency, self.power, **kwargs)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(f"Power [${self.unit}^2/Hz$]")
        plt.xscale("log")
        plt.yscale("log")
        if self.num_points_per_block:
            plt.title(f"Blocked Power spectrum (N={self.num_points_per_block})")
        else:
            plt.title("Power spectrum")
