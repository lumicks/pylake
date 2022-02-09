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
