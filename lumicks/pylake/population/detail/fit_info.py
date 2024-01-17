from dataclasses import dataclass


@dataclass(frozen=True)
class FitInfo:
    converged: bool
    n_iter: int
    bic: float
    aic: float


@dataclass(frozen=True)
class GmmFitInfo(FitInfo):
    r"""
    Fitting information for a HiddenMarkovModel

    Parameters
    ----------
    converged: bool
        Whether the model converged
    n_iter: int
        Number of iterations during the fit
    bic: float
        Bayesian Information Criterion (BIC) on fitted data

        .. math::
            BIC = k \ln{(n)} - 2 \ln{(L)}

        Where :math:`k` is the number of fitted parameters, :math:`n` is the number of observations
        (data points) and :math:`L` is the maximized value of the likelihood function
    aic: float
        Akaike Information Criterion on fitted data

        .. math::
            AIC = 2 k - 2 \ln{(L)}

        Where :math:`k` is the number of fitted parameters, :math:`n` is the number of observations
        (data points) and :math:`L` is the maximized value of the likelihood function
    lower_bound: float
        Lower bound value on the log likelihood
    """

    lower_bound: float


@dataclass(frozen=True)
class HmmFitInfo(FitInfo):
    r"""
    Fitting information for a HiddenMarkovModel

    Parameters
    ----------
    converged: bool
        Whether the model converged
    n_iter: int
        Number of iterations during the fit
    bic: float
        Bayesian Information Criterion (BIC) on fitted data

        .. math::
            BIC = k \ln{(n)} - 2 \ln{(L)}

        Where :math:`k` is the number of fitted parameters, :math:`n` is the number of observations
        (data points) and :math:`L` is the maximized value of the likelihood function
    aic: float
        Akaike Information Criterion on fitted data

        .. math::
            AIC = 2 k - 2 \ln{(L)}

        Where :math:`k` is the number of fitted parameters, :math:`n` is the number of observations
        (data points) and :math:`L` is the maximized value of the likelihood function
    log_likelihood: float
        Log Likelihood of the fitted data
    """

    log_likelihood: float
