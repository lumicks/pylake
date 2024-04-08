import numpy as np
import scipy

from lumicks.pylake.force_calibration.power_spectrum import PowerSpectrum


def estimate_driving_input_parameters(
    sample_rate, driving_data, f_drive_guess, window_factor=10, f_search=5, n_fit=1
):
    """This function finds the amplitude and frequency of a sinusoidal signal in the data.

    It windows the data by a Gaussian curve. The Fourier transform of a Gaussian is another
    Gaussian. Taking the logarithm of that then reduces it to a quadratic function which can very
    quickly be fitted [1]:

        gaussian = N * A * exp(-0.5((f - mu) / sigma) ** 2)

    Here N is the normalization constant, A is the amplitude, f is the frequency axis, mu is the
    driving frequency and sigma is the width of the Gaussian.

        log(gaussian) = log(N) + log(A) - 0.5 * ((f - mu) / sigma) ** 2

    With N being:

        1 / sqrt(2 * pi * sigma ** 2)

    Resulting in:

        log(gaussian) = -0.5 * log(2 * pi * sigma ** 2) + log(A) - 0.5 * ((f - mu) / sigma) ** 2

    Or:

        y = -0.5 * f ** 2 / sigma ** 2
            + f * mu / sigma ** 2
            - 0.5 * mu ** 2 / sigma ** 2 - 0.5 * log(2 * pi * sigma ** 2) + log(A)

        a = - 0.5 / sigma ** 2
        b = mu / sigma ** 2
        c = - 0.5 * log(2 * pi * sigma ** 2) + log(A) - 0.5 * mu ** 2 / sigma ** 2

    Which means:

        log(A) = c + 0.5 * mu ** 2 / sigma ** 2 + 0.5 * log(2 * pi * sigma ** 2)

    Or:

        A = exp(c - 0.25 * b ** 2 / a + 0.5 * log(-pi / a))

    [1] Gasior, M., & Gonzalez, J. L. (2004, November). Improving FFT frequency measurement
    resolution by parabolic and Gaussian spectrum interpolation. In AIP Conference Proceedings
    (Vol. 732, No. 1, pp. 276-285). American Institute of Physics.

    Note: This function is meant to be used for high SNR signals only.

    Parameters
    ----------
    sample_rate : int
        sample rate in Hertz
    driving_data : np.array
        data to fit
    f_drive_guess : float
        guess of the driving frequency
    window_factor : float, optional
        data is windowed in the time domain with a Gaussian of width num_points / window_factor. The
        window should be chosen such that the window has decayed sufficiently to zero (bigger than
        5 is recommended) but low enough to make sure the resulting peaks are narrow (preventing
        spectral bleed). Default value is 10.
    f_search : float, optional
        how close the target frequency is expected to be to the entered value
    n_fit : int, optional
        n_fit is how many points on either side are included in the fit (e.g. [-n_fit : n_fit + 1])

        Note: More does not necessarily mean better as the points get noisier away from the peak!
        Default is set to 1.

    Returns
    -------
    amp : float
        Amplitude of the determined oscillation.
    freq : float
        Frequency of the determined oscillation.
    amp_std : float
        Rough estimate of the amplitude error. Note that this estimate is based on a uniform
        noise distribution under the peak and may yield inaccurate results when this assumption
        is violated.
    """
    # Standard deviation of the Gaussian curve in the time domain
    num_points = len(driving_data)

    std = num_points / window_factor

    # Multiply the signal with a Gaussian window after removing any constant offset (which would
    # lead to the spectral bin belonging to the offset bleeding into higher frequencies).
    gauss_window = scipy.signal.windows.gaussian(M=num_points, std=std, sym=False)
    windowed_data = gauss_window * (driving_data - np.mean(driving_data))
    windowed_fft = np.fft.rfft(windowed_data)

    # Search near the frequency estimate.
    frequency = np.fft.rfftfreq(num_points, 1 / sample_rate)
    search_range = np.logical_and(
        frequency > f_drive_guess - f_search, frequency < f_drive_guess + f_search
    )
    max_idx = np.where(search_range)[0][0] + np.argmax(np.abs(windowed_fft[search_range]))

    # Narrow the fitting region to a few samples around the peak.
    fit_range = np.arange(max_idx - n_fit, max_idx + n_fit + 1)
    log_magnitudes = np.log(np.abs(windowed_fft[fit_range]))

    p = np.polyfit(frequency[fit_range], log_magnitudes, 2)

    if p[0] >= 0:
        raise RuntimeError(
            "Did not manage to find driving peak in spectral search range. "
            "Check whether your initial driving frequency guess is close enough to the driving "
            "frequency."
        )

    freq = -p[1] / (2.0 * p[0])  # Location of the peak of the quadratic approximation

    if freq < f_drive_guess - f_search or freq > f_drive_guess + f_search:
        raise RuntimeError(
            "Peak is outside frequency search range. Check whether your initial driving frequency "
            "guess is close enough to the driving frequency."
        )

    delta_freq = 2 / sample_rate
    # TODO: This neglects truncation effects (due to the implicit rectangular window being
    #  applied). For large values of the window_factor (>5) this is fine, but for smaller values,
    #  this can lead to systematic bias.
    amp = np.exp(p[2] - 0.25 * p[1] ** 2 / p[0] + 0.5 * np.log(-np.pi / p[0])) * delta_freq

    # Calculate the equivalent noise bandwidth for this window
    enbw = num_points * np.sum(gauss_window**2) / (np.sum(gauss_window) ** 2)

    # Rough estimate of the noise power under the quantified peak. This assumes uniform noise under
    # the peak.
    total_power = np.var(driving_data)
    amp_power = amp**2 / 2  # variance
    noise_std = np.sqrt(np.abs(total_power - amp_power))
    amp_std = enbw * noise_std / np.sqrt(num_points)

    return amp, freq, amp_std


class DrivenPower:
    def __init__(self, psd_data, sample_rate, driving_frequency, num_windows, freq_window=50.0):
        """This class is used to determine power in the driven peak."""
        self.ps = PowerSpectrum(
            psd_data, sample_rate, window_seconds=num_windows / driving_frequency
        ).in_range(max(1.0, driving_frequency - freq_window), driving_frequency + freq_window)

    def determine_power_output(self, thermal_noise):
        """Determines the driven power output in the presence of thermal background noise"""
        max_idx = np.argmax(self.ps.power)
        max_power_density = self.ps.power[max_idx]
        df = self.ps.frequency_bin_width

        # Equation 14 from [6]
        power_exp = (max_power_density - thermal_noise) * df

        # We include a contribution of the thermal noise here. This contribution should be a lot
        # smaller as sigma_thermal is given by P_thermal / sqrt(N), leading to a conservative
        # estimate.
        power_exp_err = (
            np.sqrt(self.ps._variance[max_idx] / self.ps.num_points_per_block) * df
            if self.ps._variance is not None
            else np.nan
        )
        return power_exp, power_exp_err
