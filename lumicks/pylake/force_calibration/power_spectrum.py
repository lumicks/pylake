import warnings
from copy import copy
from typing import List, Tuple

import numpy as np

from lumicks.pylake.detail.utilities import downsample


def discontiguous_plot_data(frequency, power, excluded_ranges):
    """Determine which points were plotted over a gap, and return the data to plot them

    Frequency exclusion ranges are applied prior to down-sampling. Because of this, a data point
    can end up in the gap induced by an excluded range. This function returns data where the
    gaps are explicitly shown (by inserting `np.nan` on either side of the gap) and a list of
    points that were averaged over the gap (and therefore end up inside the gap region).

    Parameters
    ----------
    frequency : np.ndarray
        Frequency axis
    power : np.ndarray
        Power spectral density values
    excluded_ranges : list of tuples of float
        Ranges that should be excluded from the plot

    Returns
    -------
    frequency : np.ndarray
        Frequency axis with nan insertions around "breaks" in the plot
    power : np.ndarray
        Power spectral densities with nan insertions around "breaks" in the plot
    freq_gap : np.ndarray
        Frequency values for the points that were averaged over a gap
    power_gap : np.ndarray
        Power spectral densities for the points that were averaged over a gap
    """
    breaks = [e[1] for e in excluded_ranges]
    break_idx = np.searchsorted(frequency, breaks, "left")

    nan_insertion_points = np.hstack((break_idx, break_idx - 1))
    return (
        np.insert(frequency, nan_insertion_points, np.nan),
        np.insert(power, nan_insertion_points, np.nan),
        frequency[break_idx - 1],
        power[break_idx - 1],
    )


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

    def __init__(
        self,
        frequency,
        power,
        sample_rate,
        total_duration,
        unit="V",
        *,
        num_points_per_block=1,
        total_samples_used=None,
        variance=None,
    ):
        """Power spectrum

        frequency : array_like
            Frequency axis
        power : array_like
            Power spectral values
        sample_rate : int
            Sample rate
        total_duration : float
            Total measurement duration
        unit : str
            Unit of the spectrum
        num_points_per_block : int
            Number of points per block of the source data
        total_samples_used : int
            Total samples used to compute FFT
        variance : array_like
            Variance of each power spectral density point
        """
        self._frequency = np.asarray(frequency)
        self._power = np.asarray(power)
        self.sample_rate = sample_rate
        self.total_duration = total_duration
        self.unit = unit

        self.total_samples_used = total_samples_used
        self._raw_variance = variance
        self._num_points_per_block = num_points_per_block

        self._downsampling_factor = 1
        self._fit_range = (
            np.nextafter(frequency.min(), -np.inf),
            np.nextafter(frequency.max(), np.inf),
        )
        self._excluded_ranges = []

    @property
    def num_points_per_block(self) -> int:
        return self._num_points_per_block * self._downsampling_factor

    @staticmethod
    def from_data(data, sample_rate, unit="V", window_seconds=None) -> "PowerSpectrum":
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

        data = data - np.mean(data)

        # Calculate power spectrum for chunks of data.
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
        scaling_factor = (2.0 / sample_rate) / num_points_per_window
        frequency = np.fft.rfftfreq(num_points_per_window, 1.0 / sample_rate)

        return PowerSpectrum(
            frequency=frequency,
            power=scaling_factor * squared_fft,
            sample_rate=sample_rate,
            total_duration=data.size / sample_rate,
            unit=unit,
            num_points_per_block=len(squared_fft_chunks),
            total_samples_used=num_points_per_window * (len(data) // num_points_per_window),
            variance=(
                scaling_factor**2 * np.var(squared_fft_chunks, axis=0)
                if len(squared_fft_chunks) > 1
                else None
            ),
        )

    @property
    def total_sampled_used(self) -> int:
        return self.total_samples_used

    @property
    def frequency_bin_width(self) -> float:
        """Returns the frequency bin width of the spectrum"""
        return self.sample_rate / self.total_samples_used * self.num_points_per_block

    def _apply_transforms(self, data, with_exclusions=True, downsample_data=True) -> np.ndarray:
        """Apply transformations to the raw spectral data

        Prior to plotting, specific sections of the spectrum are typically excluded and the data
        (both the frequency axis and power spectral density) is typically down-sampled.

        Parameters
        ----------
        data : np.ndarray
            Data to apply transformations to.
        with_exclusions : np.ndarray
            Whether to apply the frequency exclusion ranges (these carve out noise peaks and
            the active calibration peak).
        downsample_data : bool
            Should the data be down-sampled?
        """
        fit_range = (self._frequency > self._fit_range[0]) & (self._frequency <= self._fit_range[1])

        if not with_exclusions:
            return (
                downsample(data[fit_range], self._downsampling_factor, np.mean)
                if downsample_data
                else data[fit_range]
            )

        exclusion_mask = np.logical_and.reduce(
            [
                (self._frequency < f_min) | (self._frequency >= f_max)
                for f_min, f_max in self._excluded_ranges
            ]
        )

        data = data[exclusion_mask & fit_range]

        return downsample(data, self._downsampling_factor, np.mean) if downsample_data else data

    @property
    def frequency(self) -> np.ndarray:
        return self._apply_transforms(self._frequency)

    @property
    def power(self) -> np.ndarray:
        return self._apply_transforms(self._power)

    @property
    def unfiltered_frequency(self) -> np.ndarray:
        """Returns full frequency axis of the power spectrum without exclusion ranges applied."""
        return self._apply_transforms(self._frequency, with_exclusions=False)

    @property
    def unfiltered_power(self) -> np.ndarray:
        """Returns full power spectral density without exclusion ranges applied."""
        return self._apply_transforms(self._power, with_exclusions=False)

    @property
    def _variance(self) -> np.ndarray | None:
        if self._raw_variance is None:
            return None

        if self._downsampling_factor != 1:
            raise RuntimeError("Variance is only available for non-downsampled spectrum")

        return self._apply_transforms(self._raw_variance)

    def downsampled_by(self, factor, reduce=np.mean) -> "PowerSpectrum":
        """Returns a spectrum downsampled by a given factor.

        Parameters
        ----------
        factor : int
            Factor to down-sample the spectrum by.
        reduce : callable
            (Deprecated) Function to use for down-sampling the data. Only `np.mean` will be
            supported going forward.
        """
        if reduce != np.mean:
            warnings.warn(
                DeprecationWarning(
                    "Providing other reduction functions than `np.mean` is deprecated and will be "
                    "removed in a future version of Pylake"
                )
            )
            return PowerSpectrum(
                downsample(self.frequency, factor, reduce),
                downsample(self.power, factor, reduce),
                self.sample_rate,
                self.total_duration,
                self.unit,
                num_points_per_block=self.num_points_per_block * factor,
                total_samples_used=self.total_sampled_used,
                variance=None,
            )

        ba = copy(self)
        ba._downsampling_factor = ba._downsampling_factor * factor

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
        ps._excluded_ranges = self._excluded_ranges + list(excluded_ranges)
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

        if self._num_points_per_block != 1:
            raise ValueError(
                "identify_peaks only works if the power spectrum is not blocked / averaged"
            )

        if peak_cutoff <= baseline:
            raise ValueError("peak_cutoff must be greater than baseline value")

        if baseline < 0:
            raise ValueError("baseline cannot be negative")

        power = self._apply_transforms(self._power, downsample_data=False)
        frequency = self._apply_transforms(self._frequency, downsample_data=False)

        # Normalize the spectrum
        flattened_spectrum = power / model_fun(frequency)

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
        df = frequency[1] - frequency[0]
        return_val = [(frequency[x[0]], frequency[x[1]] + df) for x in exclusion_ranges]
        return return_val

    def in_range(self, frequency_min, frequency_max) -> "PowerSpectrum":
        """Returns part of the power spectrum within a given frequency range."""
        ir = copy(self)
        ir._fit_range = (
            max(self._fit_range[0], frequency_min),
            min(self._fit_range[1], frequency_max),
        )

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

        return PowerSpectrum(
            self.frequency,
            power,
            self.sample_rate,
            self.total_duration,
            self.unit,
            num_points_per_block=num_points_per_block,
            total_samples_used=self.total_samples_used,
            variance=variance,
        )

    def plot(self, *, show_excluded=False, **kwargs):
        """Plot power spectrum

        Parameters
        ----------
        show_excluded : bool
            Show ranges that were excluded from fitting.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`.
        """
        import matplotlib.pyplot as plt

        if show_excluded:
            if self._downsampling_factor > 1:
                plt.plot(self.unfiltered_frequency, self.unfiltered_power, label="_")

            for freq_min, freq_max in self._excluded_ranges:
                plt.axvspan(freq_min, freq_max, alpha=0.1)
                # It is important to draw both span and line, since cuts can be so narrow that the
                # span alone doesn't show up.
                plt.axvline(freq_min, alpha=0.1)
                plt.axvline(freq_max, alpha=0.1)

        # Inserting a np.nan for every excluded range, makes sure the plot has "breaks". It is
        # preferable over using separate plots as this won't increment the color cycle.
        frequency, power, freq_gap, power_gap = discontiguous_plot_data(
            self.frequency, self.power, self._excluded_ranges
        )

        lines = plt.plot(frequency, power, **kwargs)
        # Plot dots for points that were the result of averaging over a gap
        plt.plot(freq_gap, power_gap, linestyle="", marker=".", color=lines[0].get_color())

        plt.xlabel("Frequency [Hz]")
        plt.ylabel(f"Power [${self.unit}^2/Hz$]")
        plt.xscale("log")
        plt.yscale("log")
        if self.num_points_per_block > 1:
            plt.title(f"Blocked Power spectrum (N={self.num_points_per_block})")
        else:
            plt.title("Power spectrum")
