from dataclasses import dataclass


@dataclass(frozen=True)
class FitInfo:
    converged: bool
    n_iter: int
    bic: float
    aic: float


@dataclass(frozen=True)
class HmmFitInfo(FitInfo):
    """
    Fitting information for a HiddenMarkovModel

    Parameters
    ----------
    converged: bool
        Whether the model converged
    n_iter: int
        Number of iterations during the fit
    bic: float
        Bayesian Information Criterion on fitted data
    aic: float
        Akaike Information Criterion on fitted data
    log_likelihood: float
        Log Likelihood of the fitted data
    """

    log_likelihood: float
