import numpy as np
from scipy.optimize import curve_fit


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x: k1 * (x - x0) + y0, lambda x: k2 * (x - x0) + y0])


def fit_piecewise_linear(x, y):
    """Fits a two-segment piecewise linear function and returns the parameters.

    Parameters:
        x: array-like
            independent variable
        y: array-like
            dependent variable
    """
    # We should be able to handle cases where x moves in positive or negative direction
    difference = x[-1] - x[0]
    center = x[0] + difference / 2
    mid_point = np.nonzero(x > center)[0][0] if difference > 0 else np.nonzero(x < center)[0][0]
    x_start, y_start, x_mid, y_mid, x_end, y_end = (
        (x[0], y[0], x[mid_point], y[mid_point], x[-1], y[-1])
        if difference > 0
        else (x[-1], y[-1], x[mid_point], y[mid_point], x[0], y[0])
    )

    slope1_est = (y_mid - y_start) / (x_mid - x_start)
    slope2_est = (y_end - y_mid) / (x_end - x_mid)
    initial_guess = [x_mid, y_mid, slope1_est, slope2_est]
    pars, _ = curve_fit(piecewise_linear, x, y, initial_guess)

    return pars


def fit_sine_with_polynomial(independent, dependent, freq_guess, freq_bounds, background_degree):
    """Fit a sine wave plus polynomial background.

    We wish to fit a sine wave (with phase shift) plus a polynomial. By using a trick we can rewrite
    this equation such that we only have to optimize over 1 variable:

        amp * sin(a * x + phase) + other

    can we written as:

        amp * sin(phase) * cos(a * x) + amp * cos(phase) * sin(a * x) + other

    By absorbing the sin and cosine into the amplitude, we change variables from amplitude and
    phase, to sine amplitude and cosine amplitude. This means the problem of fitting a sine plus
    polynomial is essentially a linear one in all variables except the frequency (our parameter of
    interest). Doing this allows us to estimate this using only one estimated variable.

    Parameters
    ----------
    independent : np.ndarray
        Values for the independent variable
    dependent : np.ndarray
        Values for the dependent variable
    freq_guess : float
        Initial guess for the frequency
    freq_bounds : 2-tuple of array_like
        Bounds for the frequency guess
    background_degree : int
        Polynomial degree to use to fit the background
    """

    def sine_with_polynomial(x, frequency):
        design_matrix = np.vstack(
            (
                np.sin(2.0 * np.pi * frequency * x),
                np.cos(2.0 * np.pi * frequency * x),
                x[np.newaxis, :] ** np.arange(0, background_degree + 1)[:, np.newaxis],
            ),
        )
        ests, _, _, _ = np.linalg.lstsq(design_matrix.T, dependent, rcond=None)
        return np.sum(design_matrix.T * ests, axis=1)

    par, _ = curve_fit(sine_with_polynomial, independent, dependent, freq_guess, bounds=freq_bounds)
    return par[0], sine_with_polynomial(independent, par[0])

