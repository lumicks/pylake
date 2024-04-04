import warnings
from copy import copy
from typing import List, Tuple

import numpy as np

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

        data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError(
                f"Only 1D arrays of data are supported. You provided a {data.ndim}D array of "
                f"shape {data.shape}."
            )

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
        scaling_factor = (2.0 / sample_rate) / num_points_per_window
        self.power = scaling_factor * squared_fft

        # Store a variance for temporally blocked power spectra
        self._variance = (
            scaling_factor**2 * np.var(squared_fft_chunks, axis=0)
            if len(squared_fft_chunks) > 1
            else None
        )

        # Store metadata
        self.sample_rate = sample_rate
        self.total_duration = data.size / sample_rate
        self.num_points_per_block = len(squared_fft_chunks)
        self.total_sampled_used = num_points_per_window * self.num_points_per_block

    @property
    def frequency_bin_width(self):
        """Returns the frequency bin width of the spectrum"""
        return self.sample_rate / self.total_sampled_used * self.num_points_per_block

    def downsampled_by(self, factor, reduce=np.mean) -> "PowerSpectrum":
        """Returns a spectrum downsampled by a given factor."""
        ba = copy(self)
        ba.frequency = downsample(self.frequency, factor, reduce)
        ba.power = downsample(self.power, factor, reduce)
        ba.num_points_per_block = self.num_points_per_block * factor
        ba._variance = None  # Not supported

        return ba

    def _exclude_range(self, excluded_ranges) -> "PowerSpectrum":
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

        if self._variance is not None:
            ps._variance = ps._variance[indices]

        return ps

    def identify_peaks(
        self,
        model_fun: callable,
        *,
        peak_cutoff: float = 20.0,
        baseline: float = 1.0,
    ) -> List[Tuple[float, float]]:
        """Identify peaks in the power spectrum, based on an exponential probability model with
        rate parameter (lambda) = 1. This means that the power spectrum cannot be blocked or
        windowed. This is beta functionality. While usable, this has not yet been tested in a large
        number of different scenarios. The API can still be subject to change without any prior
        deprecation notice! If you use this functionality keep a close eye on the changelog for any
        changes that may affect your analysis.

        Parameters
        ----------
        model_fun : callable
            A function of one argument, frequency, that gives the theoretical power spectrum. The
            function is used to normalize the experimental power spectrum. Note that you can
            pass an instance of `CalibrationResults`.
        peak_cutoff: float
            Indicates what value of the normalized spectrum is deemed abnormally high. Default is
            20.0, which corresponds to a chance of about 2 in 1E9 that a peak of that magnitude
            occurs naturally in an exponential distribution with rate parameter = 1.0. The minimum
            is baseline (see below).
        baseline: float
            The baseline level a peak needs to drop down to. Lower means that exponentially more
            data is considered to be part of the peak. The default is 1.0, and the range of baseline
            is [0.0, peak_cutoff]. No fit or data smoothing is performed, the peak starts or ends at
            the first data point that satisfies the criterion.

        Returns
        -------
        frequency_ranges: list of tuples (f_start, f_stop)
            f_start is the frequency where a peak starts, and f_stop is the frequency where a peak
            ends (exclusive).

        Raises
        ------
        ValueError
            Raises a ValueError if the function is called on a PowerSpectrum object with blocking
            applied. This function only works for PowerSpectrum objects without blocking or
            windowing.
        ValueError
            Raises a ValueError when the peak_cutoff is smaller than baseline and when baseline is
            less than zero
        """

        def grab_contiguous_ranges(mask):
            # We cap the entire spectrum to make sure that we can handle spectra that begin
            # or end above the baseline in a uniform manner with ones that don't.
            capped_mask = np.diff(mask, prepend=0, append=0)
            ranges = np.nonzero(capped_mask)[0].reshape((-1, 2))
            ranges[:, 1] -= 1
            return ranges

        if self.num_points_per_block != 1:
            raise ValueError(
                "identify_peaks only works if the power spectrum is not blocked / averaged"
            )

        if peak_cutoff <= baseline:
            raise ValueError("peak_cutoff must be greater than baseline value")

        if baseline < 0:
            raise ValueError("baseline cannot be negative")

        # Normalize the spectrum
        flattened_spectrum = self.power / model_fun(self.frequency)

        baseline_mask = (flattened_spectrum >= baseline).astype("int")
        peak_mask = (flattened_spectrum > peak_cutoff).astype("int")

        peak_ranges = grab_contiguous_ranges(peak_mask)

        if not peak_ranges.size:
            return []

        baseline_ranges = grab_contiguous_ranges(baseline_mask)

        # Find start points of baseline sections (int because derivative can be negative, and bool
        # doesn't allow that).
        start_points = np.diff(baseline_mask, prepend=0) > 0

        # This allows us to look up which baseline range to grab
        baseline_indices = np.cumsum(start_points) - 1

        # Any point inside the peak range will do to identify the baseline range
        exclusion_baseline_indices = [
            baseline_indices[peak_position] for peak_position in peak_ranges[:, 0]
        ]

        # Only report unique ones
        exclusion_ranges = baseline_ranges[np.unique(exclusion_baseline_indices)]
        # Convert the indices to frequencies
        df = self.frequency[1] - self.frequency[0]
        return_val = [(self.frequency[x[0]], self.frequency[x[1]] + df) for x in exclusion_ranges]
        return return_val

    def in_range(self, frequency_min, frequency_max) -> "PowerSpectrum":
        """Returns part of the power spectrum within a given frequency range."""
        ir = copy(self)
        mask = (self.frequency > frequency_min) & (self.frequency <= frequency_max)
        ir.frequency = self.frequency[mask]
        ir.power = self.power[mask]

        if self._variance is not None:
            ir._variance = self._variance[mask]

        return ir

    def num_samples(self) -> int:
        return self.frequency.size

    def with_spectrum(self, power, num_points_per_block=1, variance=None) -> "PowerSpectrum":
        """Return a copy with a different spectrum

        Parameters
        ----------
        power : numpy.ndarray
            Vector of power spectral values
        num_points_per_block : int
            Number of points per block used to obtain power spectral values.
        variance : numpy.ndarray, optional
            Variance of the power spectrum.

        Returns
        -------
        power_spectrum : PowerSpectrum
            Power spectrum with new spectral density values.

        Raises
        ------
        ValueError
            If the power spectrum provided has a different length from the current one.
        """
        if len(power) != len(self.power):
            raise ValueError("New power spectral density vector has incorrect length")

        ps = copy(self)
        ps.power = power
        ps.num_points_per_block = num_points_per_block
        ps._variance = variance

        return ps

    def plot(self, **kwargs):
        """Plot power spectrum

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`.
        """
        import matplotlib.pyplot as plt

        plt.plot(self.frequency, self.power, **kwargs)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(f"Power [${self.unit}^2/Hz$]")
        plt.xscale("log")
        plt.yscale("log")
        if self.num_points_per_block > 1:
            plt.title(f"Blocked Power spectrum (N={self.num_points_per_block})")
        else:
            plt.title("Power spectrum")
