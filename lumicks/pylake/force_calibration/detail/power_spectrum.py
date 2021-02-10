import numpy as np
import matplotlib.pyplot as plt
import math
from copy import copy


def block_reduce(data, n_blocks, reduce=np.mean):
    """Calculates the block average of a dataset.

    For an array ``A`` of length ``N``, returns an array ``B`` of length
    ``M``, where each element of ``B`` is the average of ``q`` neighboring
    elements. ``q`` is equal to ``floor(N/M)``. This implies that if ``N*q``
    is not exactly equal to ``M``, the last partially complete window is
    thrown away by this function.
    """
    block_size = math.floor(data.size / n_blocks)
    length = block_size * n_blocks
    return reduce(np.reshape(data[:length], (-1, block_size)), axis=1)


class PowerSpectrum:
    """Power spectrum data for a time series.

    Attributes
    ----------
    f : numpy.ndarray
        Frequency values for the power spectrum. [Hz]
    P : numpy.ndarray
        Power values for the power spectrum (typically in V^2/s).
    sampling_rate : float
        The sampling rate for the original data. [Hz]
    total_duration : float
        The total duration of the original data. [seconds]
    """

    def __init__(self, data, sampling_rate, unit="V"):
        """Power spectrum

        Parameters
        ----------
        data : numpy.ndarray
            Data from which to calculate a power spectrum.
        sampling_rate : float
            Sampling rate at which this data was acquired.
        unit : str
            Units the data is in (default: V).
        """
        self.unit = unit
        data = data - np.mean(data)

        # Calculate power spectrum.
        fft = np.fft.rfft(data)
        self.f = np.fft.rfftfreq(data.size, 1.0 / sampling_rate)
        self.P = (1.0 / sampling_rate) * np.square(np.abs(fft)) / data.size

        # Store metadata
        self.sampling_rate = sampling_rate
        self.total_duration = data.size / sampling_rate
        self.num_points_per_block = 1

    def as_dict(self):
        """"Returns a representation of the PowerSpectrum suitable for serialization"""
        return {"f": self.f.tolist(), "P": self.P.tolist()}

    def block_averaged(self, num_blocks=2000):
        """Returns a block-averaged power spectrum.

        See Also
        --------
        block_average
        """
        ba = copy(self)
        ba.f = block_reduce(self.f, num_blocks)
        ba.P = block_reduce(self.P, num_blocks)
        ba.num_points_per_block = self.num_points_per_block * math.floor(self.P.size / num_blocks)

        return ba

    def in_range(self, f_min, f_max):
        """Returns part of the power spectrum within a given frequency range."""
        ir = copy(self)
        mask = (self.f > f_min) & (self.f <= f_max)
        ir.f = self.f[mask]
        ir.P = self.P[mask]
        return ir

    def num_samples(self):
        return self.f.size

    def with_spectrum(self, power, num_points_per_block=1):
        """Return a copy with a different spectrum"""
        assert len(power) == len(self.P), "Power has incorrect length"

        ps = copy(self)
        ps.P = power
        ps.num_points_per_block = num_points_per_block

        return ps

    def plot(self):
        """Plot power spectrum"""
        plt.plot(self.f, self.P)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(f"Power [${self.unit}^2/Hz$]")
        plt.xscale("log")
        plt.yscale("log")
        if self.num_points_per_block:
            plt.title(f"Blocked Power spectrum (N={self.num_points_per_block})")
        else:
            plt.title("Power spectrum")
