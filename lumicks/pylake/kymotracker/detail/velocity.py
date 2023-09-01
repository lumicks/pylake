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

    @property
    def converged(self):
        return self.exitflag == "converged"


def fit_piecewise_continuous(x, y, n_breakpoints, tol=1e-5, max_iter=30, n_restarts=100):
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

    results = [_optimize_breakpoints(x, y, n_breakpoints, tol, max_iter) for _ in range(n_restarts)]
    filtered_results = [result for result in results if result.converged]

    # find best converged fit, fallback to un-converged if necessary
    return min(filtered_results if len(filtered_results) else results, key=lambda r: r.rss)


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


def _optimize_breakpoints(x, y, n_breakpoints, tol, max_iter):
    r"""
    Optimization of breakpoint locations for a continuous piecewise linear model

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
    """
    n_samples = len(x)
    n_coeffs = 2 + 2 * n_breakpoints

    def fit_iteration(breakpoints):
        r"""
        Single iteration of the breakpoint estimation algorithm, solving Eq 7. corresponding
        to linearized form of piecewise continuous model, with nonlinear breakpoints term:

        .. math::

            \alpha x + \beta U + \gamma V

        where :math:`\alpha` is the slope of the first section, :math:`\beta` models the
        difference-in-slopes of the subsequent sections, and :math:`\gamma` is a reparameterization
        of the breakpoints :math:`\psi`.

        Given the current breakpoints guess :math:`\psi^{(0)}` and the ML estimates of above
        parameters using OLS, the updated breakpoints can be calculated as:

        .. math::

            \psi = \frac{\gamma}{\beta} + \psi^{(0)}
        """
        diff_term = x - breakpoints[:, np.newaxis]
        heavi = np.vstack([np.heaviside(d, -1) for d in diff_term])
        u = diff_term * heavi
        v = -heavi
        design_matrix = np.vstack((np.ones(x.size), x, u, v)).T

        xtx = np.linalg.pinv(np.matmul(design_matrix.T, design_matrix))
        coeffs = np.dot(
            xtx,
            np.dot(design_matrix.T, y),
        )

        fit = np.dot(design_matrix, coeffs)
        residuals = fit - y
        rss = np.sum(residuals**2)

        intercept, alpha, *others = coeffs
        beta, gamma = np.reshape(others, (2, n_breakpoints))

        cov = (rss / (n_samples - n_coeffs)) * xtx

        return intercept, alpha, beta, gamma, cov, rss

    exitflag = "max iterations reached"
    breakpoints = np.sort(np.random.uniform(np.min(x), np.max(x), n_breakpoints))

    for _ in range(max_iter):
        intercept, alpha, beta, gamma, cov, rss = fit_iteration(breakpoints)

        if np.all(gamma < tol):
            exitflag = "converged"
            break

        # update breakpoints and check in bounds
        breakpoints = gamma / beta + breakpoints
        if np.any(breakpoints < np.min(x)) or np.any(breakpoints > np.max(x)):
            exitflag = "breakpoints out of bounds"
            break

    # evaluate one more time to ensure that breakpoints are sorted
    breakpoints = np.sort(breakpoints)
    intercept, alpha, beta, gamma, cov, rss = fit_iteration(breakpoints)
    bic = n_samples * np.log(rss / n_samples) + n_coeffs * np.log(n_samples)

    # convert optimization coefficients to segment linear coefficients
    # intercept remains, just calculate standard error
    intercept_std = cov[0, 0]

    # slopes
    n_terms = n_breakpoints + 1
    slope_terms = np.hstack((alpha, beta))
    slope_block = cov[1 : n_terms + 1, 1 : n_terms + 1]
    slopes = np.array([np.sum(slope_terms[:j]) for j in range(1, n_terms + 1)])
    slopes_std = np.array([np.sqrt(np.sum(slope_block[:j, :j])) for j in range(1, n_terms + 1)])

    # breakpoints
    bg_block = cov[2:, 2:]
    beta_var, gamma_var = np.reshape(np.diag(bg_block), (2, n_breakpoints))
    bg_cov = np.diag(bg_block, k=n_breakpoints)
    bp_std = [
        (gv + bv * (g / b) ** 2 + 2 * (g / b) * bgcv) / (b**2)
        for b, g, bv, gv, bgcv in zip(beta, gamma, beta_var, gamma_var, bg_cov)
    ]

    return PiecewiseModel(
        intercept, slopes, breakpoints, intercept_std, slopes_std, bp_std, rss, bic, exitflag
    )
