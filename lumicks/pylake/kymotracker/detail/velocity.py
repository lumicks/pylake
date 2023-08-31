import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PiecewiseModel:
    """Piecewise Continuous Linear Model.

    Parameters
    ----------
    intercept : float
        y-intercept
    slopes : List[float]
        list of segment slopes
    breakpoints : List[float]
        list of breakpoints
    intercept_std : float
        estimate of intercept standard error
    slopes_std : List[float]
        estimate of the slope standard erros
    breakpoints : List[float]
        estimate of the breakpoint standard errors
    rss : float
        residual sum of squares
    bic : float
        Bayesian Information Criterion
    exitflag : str
        optimization exit condition
    """

    intercept: float
    slopes: List[float]
    breakpoints: List[float]
    intercept_std: float = field(repr=False)
    slopes_std: List[float] = field(repr=False)
    breakpoints_std: List[float] = field(repr=False)
    rss: float
    bic: float
    exitflag: str


def fit_piecewise_continuous(x, y, n_breakpoints, tol=1e-5, max_iter=30, n_restarts=1):
    """Fit a piecewise continuous model to data.

    If the number of breakpoints requested is 0, simply returns a linear fit.
    For a requested number of breakpoints, uses the method from [1]_ to optimise the breakpoint
    positions along with segement slopes and initial intercept.

    Parameters
    ----------
    x : np.ndarray
        Dependent variable (time).
    y : np.ndarray
        Independent variable (position).
    n_breakpoints : int
        Number of requested breakpoints.
    tol : float
        Convergence tolerance. Ignored if `n_breakpoints == 0`.
    max_iter : int
        Maximum number of iterations allowed in the optimization algorithm. Ignored if
        `n_breakpoints == 0`.
    n_restarts : int
        Number of times to run the optimization algorithm with breakpoint initial guesses drawn
        from a uniform distribution. Ignored if `n_breakpoints == 0`.

    Returns
    -------
    PiecewiseModel
        Parameters and optimization statistics for piecewise linear fit.

    References
    ----------
    .. [1] Muggeo, V. (2003) Estimating Regression Models with Unknown Break-Points. Statistics in
           Medicine, 22(19), 3055-3071.
    """

    if n_breakpoints < 0:
        raise ValueError(f"Number of requested breakpoints must be >= 0, got {n_breakpoints}.")
    elif n_breakpoints == 0:
        return _optimize_linear(x, y)
    else:
        # todo: implement algorithm
        pass


def _optimize_linear(x, y):
    """OLS linear regression of x and y.

    Parameters
    ----------
    x : np.ndarray
        Dependent variable (time).
    y : np.ndarray
        Independent variable (position).
    """
    n_coeffs = 2
    n_samples = len(x)

    design_matrix = np.vstack((np.ones(n_samples), x)).T
    xtx = np.linalg.pinv(np.matmul(design_matrix.T, design_matrix))
    coeffs = np.dot(
        xtx,
        np.dot(design_matrix.T, y),
    )

    fit = np.dot(design_matrix, coeffs)
    residuals = fit - y
    rss = np.sum(residuals**2)
    bic = n_samples * np.log(rss / n_samples) + n_coeffs * np.log(n_samples)

    intercept, slope = coeffs

    cov = (rss / (n_samples - n_coeffs)) * xtx
    intercept_std, slope_std = np.sqrt(np.diag(cov))

    return PiecewiseModel(
        intercept, [slope], [], intercept_std, [slope_std], [], rss, bic, "converged"
    )
