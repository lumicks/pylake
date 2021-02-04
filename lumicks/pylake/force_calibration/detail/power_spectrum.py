import numpy as np
import math


def block_average(data, n_blocks):
    """Calculates the block average of a dataset.

    For an array ``A`` of length ``N``, returns an array ``B`` of length
    ``M``, where each element of ``B`` is the average of ``q`` neighboring
    elements. ``q`` is equal to ``floor(N/M)``. This implies that if ``N*q``
    is not exactly equal to ``M``, the last partially complete window is
    thrown away by this function.
    """
    block_size = math.floor(data.size / n_blocks)
    length = block_size * n_blocks
    return np.mean(np.reshape(data[:length], (-1, block_size)), axis=1)


def block_average_std(data, n_blocks):
    """Calculates the block standard deviation of a dataset.

    Works as `block_average`, but calculates the standard deviation
    instead of the average.

    See Also
    --------
    block_average
    """
    block_size = math.floor(data.size / n_blocks)
    length = block_size * n_blocks
    return np.std(np.reshape(data[:length], (-1, block_size)), axis=1)


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
    T_measure : float
        The total duration of the original data. [seconds]
    """

    def __init__(self, data=None, sampling_rate=None):
        """Constructor

        If neither parameter is given, an empty object is created.

        Parameters
        ----------
        data : numpy.ndarray, optional
            Data from which to calculate a power spectrum.
        sampling_rate : float, optional
        """
        if data is not None:
            # Initialize from raw sensor data.
            assert sampling_rate is not None

            # Subtract average position.
            data = data - np.mean(data)

            # Calculate power spectrum.
            fft = np.fft.rfft(data)
            self.f = np.fft.rfftfreq(data.size, 1.0 / sampling_rate)
            self.P = (1.0 / sampling_rate) * np.square(np.abs(fft)) / data.size

            # Store metadata
            self.sampling_rate = sampling_rate
            self.T_measure = data.size / sampling_rate
        else:
            # Initialize empty object.
            self.f = None
            self.P = None
            self.sampling_rate = None
            self.T_measure = None

    def as_dict(self):
        """"Returns a representation of the PowerSpectrum suitable for serialization"""
        return {"f": self.f.tolist(), "P": self.P.tolist()}

    def block_averaged(self, n_blocks=2000):
        """Returns a block-averaged power spectrum.

        See Also
        --------
        block_average
        """
        ba = PowerSpectrum()
        ba.f = block_average(self.f, n_blocks)
        ba.P = block_average(self.P, n_blocks)
        ba.sampling_rate = self.sampling_rate
        ba.T_measure = self.T_measure
        return ba

    def in_range(self, f_min, f_max):
        """Returns part of the power spectrum within a given frequency range."""
        ir = PowerSpectrum()
        ir.f = self.f[(self.f > f_min) & (self.f <= f_max)]
        ir.P = self.P[(self.f > f_min) & (self.f <= f_max)]
        ir.sampling_rate = self.sampling_rate
        ir.T_measure = self.T_measure
        return ir

    def n_samples(self):
        return self.f.size
