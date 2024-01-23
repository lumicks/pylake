from dataclasses import dataclass


@dataclass(frozen=True)
class PopulationFitInfo:
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
        log of the maximized value of the likelihood function of the fitted data
    """
    converged: bool
    n_iter: int
    bic: float
    aic: float
    log_likelihood: float
