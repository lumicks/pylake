import numpy as np
from typing import Dict, Tuple, Union
from scipy.special import logsumexp
from scipy.optimize import minimize
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from deprecated.sphinx import deprecated
from lumicks.pylake.fitting.parameters import Parameter, Params
from lumicks.pylake.fitting.profile_likelihood import ProfileLikelihood1D


@dataclass(frozen=True)
class DwelltimeBootstrap:
    """Bootstrap distributions for a dwelltime model.

    This class should be initialized using :meth:`lk.DwelltimeModel.calculate_bootstrap()
    <lumicks.pylake.DwelltimeModel.calculate_bootstrap>` and should not be constructed manually.

    .. warning::

        This is early access alpha functionality. While usable, this has not yet been tested in a large number of
        different scenarios. The API can still be subject to change without any prior deprecation notice! If you
        use this functionality keep a close eye on the changelog for any changes that may affect your analysis.

    Attributes
    ----------
    model : :class:`~lumicks.pylake.DwelltimeModel`
        Original model sampled for the bootstrap distribution
    amplitude_distributions : np.ndarray
        Array of sample optimized amplitude parameters; shape is [number of components, number of samples]
    lifetime_distributions : np.ndarray
        Array of sample optimized lifetime parameters; shape is [number of components, number of samples]

    Raises
    ------
    ValueError
        If the number of amplitude samples isn't the same as the number of lifetime samples.
        If the number of parameters isn't the same as the number of components.
    """

    model: "DwelltimeModel"
    amplitude_distributions: np.ndarray = field(repr=False)
    lifetime_distributions: np.ndarray = field(repr=False)

    def __post_init__(self):
        if self.amplitude_distributions.shape[1] != self.lifetime_distributions.shape[1]:
            raise ValueError(
                f"Number of amplitude samples ({self.amplitude_distributions.shape[1]}) should be "
                f"the same as number of lifetime samples ({self.lifetime_distributions.shape[1]})."
            )

        if any(
            arr.shape[0] != self.model.n_components
            for arr in (self.amplitude_distributions, self.lifetime_distributions)
        ):
            raise ValueError(
                "Number of parameters should be the same as the number of components "
                f"({self.model.n_components})."
            )

    @classmethod
    def _from_dwelltime_model(cls, optimized, iterations):
        """Construct bootstrap distributions for parameters from an optimized :class:`~lumicks.pylake.DwelltimeModel`.

        For each iteration, a dataset is randomly selected (with replacement) with the same
        size as the data used to optimize the model. Model parameters are then optimized
        for this new sampled dataset.

        Parameters
        ----------
        optimized : DwelltimeModel
            optimized model results
        iterations : int
            number of iterations (random samples) to use for the bootstrap
        """
        samples = DwelltimeBootstrap._sample(optimized, iterations)
        return cls(optimized, samples[: optimized.n_components], samples[optimized.n_components :])

    def extend(self, iterations):
        """Extend the distribution by additional sampling iterations.

        Parameters
        ----------
        iterations : int
            number of iterations (random samples) to add to the bootstrap distribution
        """
        new_samples = DwelltimeBootstrap._sample(self.model, iterations)
        n_components = self.model.n_components
        new_amplitudes = new_samples[:n_components]
        new_lifetimes = new_samples[n_components:]

        return DwelltimeBootstrap(
            self.model,
            np.hstack([self.amplitude_distributions, new_amplitudes]),
            np.hstack([self.lifetime_distributions, new_lifetimes]),
        )

    @staticmethod
    def _sample(optimized, iterations) -> np.ndarray:
        """Calculate bootstrap samples

        Parameters
        ----------
        optimized : DwelltimeModel
            An optimized DwellTimeModel to start from.
        iterations : int
            Number of samples to generate.
        """

        n_data = optimized.dwelltimes.size
        samples = np.empty((optimized._parameters.size, iterations))
        for itr in range(iterations):
            sample = np.random.choice(optimized.dwelltimes, size=n_data, replace=True)
            result, _ = _exponential_mle_optimize(
                optimized.n_components,
                sample,
                *optimized._observation_limits,
                initial_guess=optimized._parameters,
                options=optimized._optim_options,
            )
            samples[:, itr] = result

        return samples

    @property
    def n_samples(self):
        """Number of samples in the bootstrap."""
        return self.amplitude_distributions.shape[1]

    @property
    def n_components(self):
        """Number of components in the model."""
        return self.model.n_components

    @deprecated(
        reason=(
            "This method will be removed in a future release. Use `DwelltimeBootstrap.get_interval() "
            "to obtain the `1-alpha` interval."
        ),
        action="always",
        version="0.13.3",
    )
    def calculate_stats(self, key, component, alpha=0.05):
        data = getattr(self, f"{key}_distributions")[component]
        mean = np.mean(data)
        interval = self.get_interval(key, component, alpha)
        return mean, interval

    def get_interval(self, key, component, alpha=0.05):
        """Calculate the `1-alpha` interval of the bootstrap distribution for a specified parameter.

        *NOTE*: the `100*(1-alpha)` % confidence intervals calculated here correspond to the
        `alpha/2` and `1-(alpha/2)` quantiles of the distribution. For distributions
        which are not well approximated by a normal distribution these values are not reliable
        confidence intervals.

        Parameters
        ----------
        key : {'amplitude', 'lifetime'}
            name of the parameter to be analyzed
        component : int
            index of the component to be analyzed
        alpha : float
            confidence intervals are calculated as 100*(1-alpha)%
        """
        if key not in ("amplitude", "lifetime"):
            raise KeyError("key must be either 'amplitude' or 'lifetime'")

        data = getattr(self, f"{key}_distributions")[component]
        lower = np.quantile(data, alpha / 2)
        upper = np.quantile(data, 1 - (alpha / 2))
        return lower, upper

    @deprecated(
        reason=(
            "This method has been renamed to more closely match its behavior. Use "
            ":meth:`DwelltimeBootstrap.hist()` instead."
        ),
        action="always",
        version="0.13.3",
    )
    def plot(self, alpha=0.05, n_bins=25, hist_kwargs=None, span_kwargs=None, line_kwargs=None):
        self.hist(
            n_bins=n_bins,
            alpha=alpha,
            hist_kwargs=hist_kwargs,
            span_kwargs=span_kwargs,
            line_kwargs=line_kwargs,
        )

    def hist(self, *, n_bins=25, alpha=0.05, hist_kwargs=None, span_kwargs=None, line_kwargs=None):
        """Plot the bootstrap distributions for the parameters of a model.

        Parameters
        ----------
        n_bins : int
            number of bins in the histogram
        alpha : float
            confidence intervals are calculated as 100*(1-alpha)%
        hist_kwargs : dict
            dictionary of plotting `kwargs` applied to histogram
        span_kwargs : dict
            dictionary of plotting `kwargs` applied to the patch indicating the area
            spanned by the confidence intervals
        line_kwargs : dict
            dictionary of plotting `kwargs` applied to the line indicating the
            distribution means
        """
        hist_kwargs = {"facecolor": "#c5c5c5", "edgecolor": "#888888", **(hist_kwargs or {})}
        span_kwargs = {"facecolor": "tab:red", "alpha": 0.3, **(span_kwargs or {})}
        line_kwargs = {"color": "k", **(line_kwargs or {})}

        def plot_axes(data, key, component, use_index):
            plt.hist(data, bins=n_bins, **hist_kwargs)
            mean = getattr(self.model, f"{key}s")[component]
            lower, upper = self.get_interval(key, component, alpha)
            plt.axvspan(lower, upper, **span_kwargs)
            plt.axvline(mean, **line_kwargs)
            plt.xlabel(f"{key}" if key == "amplitude" else f"{key} (sec)")
            plt.ylabel("counts")

            label = "a" if key == "amplitude" else r"\tau"
            unit = "" if key == "amplitude" else "sec"
            prefix = rf"${label}_{component+1}$" if use_index else rf"${label}$"
            plt.title(f"{prefix} = {mean:0.2g} ({lower:0.2g}, {upper:0.2g}) {unit}")

        if self.n_components == 1:
            data = self.lifetime_distributions.squeeze()
            plot_axes(data, "lifetime", 0, False)
        else:
            for component in range(self.n_components):
                for column, key in enumerate(("amplitude", "lifetime")):
                    data = getattr(self, f"{key}_distributions")[component]
                    column += 1
                    plt.subplot(self.n_components, 2, 2 * component + column)
                    plot_axes(data, key, component, True)

        plt.tight_layout()


class DwelltimeModel:
    """Exponential mixture model optimization for dwelltime analysis.

    .. warning::

        This is early access alpha functionality. While usable, this has not yet been tested in a large number of
        different scenarios. The API can still be subject to change without any prior deprecation notice! If you
        use this functionality keep a close eye on the changelog for any changes that may affect your analysis.

    Parameters
    ----------
    dwelltimes : numpy.ndarray
        observations on which the model was trained. *Note: the units of the optimized lifetime
        will be in the units of the dwelltime data. If the dwelltimes are calculated as the number
        of frames, these then need to be multiplied by the frame time in order to obtain the
        physically relevant parameter.*
    n_components : int
        number of components in the model.
    min_observation_time : float
        minimum experimental observation time
    max_observation_time : float
        maximum experimental observation time.
    tol : float
        The tolerance for optimization convergence. This parameter is forwarded as the `ftol`
        argument to :func:`scipy.optimize.minimize(method="SLSQP") <scipy.optimize.minimize()>`.
    max_iter : int
        The maximum number of iterations to perform. This parameter is forwarded as the `maxiter`
        argument to :func:`scipy.optimize.minimize(method="SLSQP") <scipy.optimize.minimize()>`.
    """

    def __init__(
        self,
        dwelltimes,
        n_components=1,
        *,
        min_observation_time=0,
        max_observation_time=np.inf,
        tol=None,
        max_iter=None,
    ):
        self.n_components = n_components
        self.dwelltimes = dwelltimes

        self._observation_limits = (min_observation_time, max_observation_time)
        self._optim_options = {
            key: value
            for key, value in zip(("ftol", "maxiter"), (tol, max_iter))
            if value is not None
        }

        self._parameters, self._log_likelihood = _exponential_mle_optimize(
            n_components,
            dwelltimes,
            min_observation_time,
            max_observation_time,
            options=self._optim_options,
        )
        # TODO: remove with deprecation
        self._bootstrap = DwelltimeBootstrap(
            self, np.empty((n_components, 0)), np.empty((n_components, 0))
        )

    @property
    @deprecated(
        reason=(
            "This property will be removed in a future release. Use the class "
            ":class:`population.dwelltime.DwelltimeBootstrap` returned from :meth:`calculate_bootstrap` instead."
        ),
        action="always",
        version="0.13.3",
    )
    def bootstrap(self):
        """Bootstrap distribution."""

        if self._bootstrap.n_samples == 0:
            raise RuntimeError(
                "The bootstrap distribution is currently empty. Use `DwelltimeModel.calculate_bootstrap()` "
                "to sample a distribution before attempting downstream analysis."
            )

        return self._bootstrap

    @property
    def amplitudes(self):
        """Fractional amplitude of each model component."""
        return self._parameters[: self.n_components]

    @property
    def lifetimes(self):
        """Lifetime parameter (in time units) of each model component."""
        return self._parameters[self.n_components :]

    @property
    def rate_constants(self):
        """First order rate constant (units of per time) of each model component."""
        return 1 / self.lifetimes

    @property
    def log_likelihood(self):
        return self._log_likelihood

    @property
    def aic(self) -> float:
        r"""Akaike Information Criterion

        .. math::

            \mathrm{AIC} = 2 k - 2 \ln{(L)}

        Where :math:`k` is the number of parameters minus the number of equality constraints and
        :math:`L` is the maximized value of the likelihood function [1]_.

        The emphasis of this criterion is future prediction. It does not lead to consistent model
        selection and is more prone to over-fitting than the Bayesian Information Criterion.

        References
        ----------
        .. [1] Cavanaugh, J.E., 1997. Unifying the derivations for the Akaike and corrected Akaike
               information criteria. Statistics & Probability Letters, 33(2), pp.201-208.
        """
        k = 2 * self.n_components - 1
        return 2 * k - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        r"""Bayesian Information Criterion

        .. math::

            \mathrm{BIC} = k \ln{(n)} - 2 \ln{(L)}

        Where :math:`k` is the number of parameters minus the number of equality constraints,
        :math:`n` is the number of observations (data points), and :math:`L` is the maximized
        value of the likelihood function.

        The emphasis of the BIC is put on parsimonious models. As such it is less prone to
        over-fitting. Selection via BIC leads to a consistent model selection procedure, meaning
        that as the number of data points tends to infinity, BIC will select the true model
        assuming the true model is in the set of models.
        """
        k = 2 * self.n_components - 1
        n = self.dwelltimes.size  # number of observations
        return k * np.log(n) - 2 * self.log_likelihood

    def calculate_bootstrap(self, iterations=500):
        """Calculate a bootstrap distribution for the model.

        Parameters
        ----------
        iterations : int
            Number of iterations to sample for the distribution.
        """
        bootstrap = DwelltimeBootstrap._from_dwelltime_model(self, iterations)
        # TODO: remove with deprecation
        self._bootstrap = bootstrap
        return bootstrap

    def profile_likelihood(
        self,
        *,
        confidence_level=0.95,
        num_steps=150,
        min_chi2_step=0.01,
        max_chi2_step=0.05,
        verbose=False,
    ) -> Dict[str, ProfileLikelihood1D]:
        """Calculate a likelihood profile.

        This method traces an optimal path through parameter space in order to estimate parameter
        confidence intervals. It iteratively performs a step for the profiled parameter, then
        fixes that parameter and re-optimizes all the other parameters [2]_ [3]_.

        Parameters
        ----------
        confidence_level : int
            Confidence level to compute confidence interval for, default: 0.95.
        num_steps: integer
            Number of steps to take, default: 150.
        min_chi2_step: float
            Minimal desired step in terms of chi squared change prior to re-optimization. When the
            step results in a fit change smaller than this threshold, the step-size will be
            increased, default: 0.01.
        max_chi2_step: float
            Minimal desired step in terms of chi squared change prior to re-optimization. When the
            step results in a fit change bigger than this threshold, the step-size will be reduced.
            Default: 0.05.
        verbose: bool
            Controls the verbosity of the output. Default: False.

        References
        ----------
        .. [2] Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmüller, U.,
               & Timmer, J. (2009). Structural and practical identifiability analysis of partially
               observed dynamical models by exploiting the profile likelihood. Bioinformatics,
               25(15), 1923-1929.
        .. [3] Maiwald, T., Hass, H., Steiert, B., Vanlier, J., Engesser, R., Raue, A., Kipkeew,
               F., Bock, H.H., Kaschek, D., Kreutz, C. and Timmer, J., 2016. Driving the model to
               its limit: profile likelihood based model reduction. PloS one, 11(9).
        """

        def model_func(params, fixed):
            """Lower bound and upper bound are handled by _exponential_mle_optimize itself"""
            return _exponential_mle_optimize(
                self.n_components,
                self.dwelltimes,
                min_observation_time=self._observation_limits[0],
                max_observation_time=self._observation_limits[1],
                initial_guess=params,
                options=self._optim_options,
                fixed_param_mask=fixed,
            )

        def fit_func(params, lb, ub, fitted):
            # Lower and upper bounds are handled by the model internally.
            return model_func(params, np.logical_not(fitted))[0]

        # Pack parameters
        keys = [f"amplitude {idx}" for idx in range(self.n_components)] + [
            f"lifetime {idx}" for idx in range(self.n_components)
        ]
        bounds = _exponential_mle_bounds(self.n_components, *self._observation_limits)
        parameters = Params(
            **{
                key: Parameter(param, lower_bound=lb, upper_bound=ub)
                for key, param, (lb, ub) in zip(keys, self._parameters, bounds)
            }
        )

        def calculate_profile(param):
            profile = ProfileLikelihood1D(
                param,
                num_dof=1,
                min_chi2_step=min_chi2_step,
                max_chi2_step=max_chi2_step,
                confidence_level=confidence_level,
            )

            def trial(params):
                """Get log likelihood for particular parameter vector"""
                # These dwelltime models are subject to internal constraints that are only ensured
                # during optimization. This means that we have to call the optimization procedure
                # with the parameter we're profiling held to a fixed value to obtain a value for the
                # chi-squared function at the trial point.
                fixed_params = [name == param for name in parameters.keys()]
                return -model_func(params, fixed_params)[1]

            profile._extend_profile(trial, fit_func, parameters, num_steps, True, verbose)
            profile._extend_profile(trial, fit_func, parameters, num_steps, False, verbose)
            return profile

        return {
            param: calculate_profile(param)
            for param in ([keys[-1]] if self.n_components == 1 else keys)
        }

    def pdf(self, x):
        """Probability Distribution Function (states as rows).

        Parameters
        ----------
        x : np.array
            array of independent variable values at which to calculate the PDF.
        """
        return np.exp(
            exponential_mixture_log_likelihood_components(
                self.amplitudes, self.lifetimes, x, *self._observation_limits
            )
        )

    @deprecated(
        reason=(
            "This method has been renamed to more closely match its behavior. Use "
            ":meth:`DwelltimeModel.hist()` instead."
        ),
        action="always",
        version="0.12.0",
    )
    def plot(
        self,
        n_bins=25,
        bin_spacing="linear",
        hist_kwargs=None,
        component_kwargs=None,
        fit_kwargs=None,
        xscale=None,
        yscale=None,
    ):
        self.hist(n_bins, bin_spacing, hist_kwargs, component_kwargs, fit_kwargs, xscale, yscale)

    def hist(
        self,
        n_bins=25,
        bin_spacing="linear",
        hist_kwargs=None,
        component_kwargs=None,
        fit_kwargs=None,
        xscale=None,
        yscale=None,
    ):
        """Plot the dwelltime distribution histogram and overlayed model density.

        Parameters
        ----------
        n_bins : int
            number of bins in the histogram
        bin_spacing : {"log", "linear"}
            determines how bin edges are spaced apart
        hist_kwargs : Optional[dict]
            dictionary of plotting kwargs applied to histogram
        component_kwargs : Optional[dict]
            dictionary of plotting kwargs applied to the line plot for each component
        fit_kwargs : Optional[dict]
            dictionary of plotting kwargs applied to line plot for the total fit
        xscale : {"log", "linear", None}
            scaling for the x-axis; when `None` default is "linear"
        yscale : {"log", "linear", None}
            scaling for the y-axis; when `None` default is same as `bin_spacing`
        """
        if bin_spacing == "log":
            scale = np.logspace
            limits = (np.log10(self.dwelltimes.min()), np.log10(self.dwelltimes.max()))
            xscale = "linear" if xscale is None else xscale
            yscale = "log" if yscale is None else yscale
        elif bin_spacing == "linear":
            scale = np.linspace
            limits = (self.dwelltimes.min(), self.dwelltimes.max())
            xscale = "linear" if xscale is None else xscale
            yscale = "linear" if yscale is None else yscale
        else:
            raise ValueError("spacing must be either 'log' or 'linear'")

        bins = scale(*limits, n_bins)
        centers = bins[:-1] + (bins[1:] - bins[:-1]) / 2

        hist_kwargs = {"facecolor": "#cdcdcd", "edgecolor": "#aaaaaa", **(hist_kwargs or {})}
        component_kwargs = {"marker": "o", "ms": 3, **(component_kwargs or {})}
        fit_kwargs = {"color": "k", **(fit_kwargs or {})}

        components = self.pdf(centers)

        def label_maker(a, t, n):
            if self.n_components == 1:
                amplitude = ""
                lifetime_label = r"$\tau$"
            else:
                amplitude = f"($a_{n}$ = {a:0.2g}) "
                lifetime_label = rf"$\tau_{n}$"
            return f"{amplitude}{lifetime_label} = {t:0.2g} sec"

        # plot histogram
        density, _, _ = plt.hist(self.dwelltimes, bins=bins, density=True, **hist_kwargs)
        # plot individual components
        for n in range(self.n_components):
            label = label_maker(self.amplitudes[n], self.lifetimes[n], n + 1)
            plt.plot(centers, components[n], label=label, **component_kwargs)
        # plot total fit
        label = r"$\ln \mathcal{L} $" + f"= {self.log_likelihood:0.3f}"
        plt.plot(centers, np.sum(components, axis=0), label=label, **fit_kwargs)

        # rearrange legend entries so that total fit is first
        legend_components = [[c[-1], *c[:-1]] for c in plt.gca().get_legend_handles_labels()]
        plt.legend(*legend_components, loc="upper right")

        # format axes
        plt.xscale(xscale)
        plt.yscale(yscale)
        if yscale == "log":
            ylim = (np.min(density[density != 0] * 0.5), np.max(density[density != 0] * 1.5))
            plt.ylim(ylim)
        plt.ylabel("density")
        plt.xlabel("dwelltime (sec)")

        plt.tight_layout()


def exponential_mixture_log_likelihood_components(
    amplitudes, lifetimes, t, min_observation_time, max_observation_time
):
    """Calculate each component of the log likelihood of an exponential mixture distribution.

    The full log likelihood for a single observation is given by:
        log(L) = log( sum_i( component_i ) )

    with the output of this function being log(component_i) defined as:
        log(component_i) = log(a_i) - log(N) + log(tau_i) - t/tau_i

    where a_i and tau_i are the amplitude and lifetime of component i and N is a normalization
    factor that takes into account the minimum and maximum observation times of the experiment:
        N = sum_i { a_i * [ exp(-t_min / tau_i) - exp(-t_max / tau_i) ] }

    Therefore, the full log likelihood is calculated from the output of this function by applying
    logsumexp(output, axis=0) where the summation is taken over the components.

    Parameters
    ----------
    amplitudes : array_like
        fractional amplitude parameters for each component
    lifetimes : array_like
        lifetime parameters for each component in seconds
    t : array_like
        dwelltime observations in seconds
    min_observation_time : float
        minimum observation time in seconds
    max_observation_time : float
        maximum observation time in seconds
    """
    amplitudes = amplitudes[:, np.newaxis]
    lifetimes = lifetimes[:, np.newaxis]
    t = t[np.newaxis, :]

    norm_factor = np.log(amplitudes) + np.log(
        np.exp(-min_observation_time / lifetimes) - np.exp(-max_observation_time / lifetimes)
    )
    log_norm_factor = logsumexp(norm_factor, axis=0)

    return -log_norm_factor + np.log(amplitudes) - np.log(lifetimes) - t / lifetimes


def exponential_mixture_log_likelihood(params, t, min_observation_time, max_observation_time):
    """Calculate the log likelihood of an exponential mixture distribution.

    The full log likelihood for a single observation is given by:
        log(L) = log( sum_i( exp( log(component_i) ) ) )

    where log(component_i) is output from `exponential_mixture_log_likelihood_components()`

    Parameters
    ----------
    params : array_like
        array of model parameters (amplitude and lifetime per component)
    t : array_like
        dwelltime observations in seconds
    min_observation_time : float
        minimum observation time in seconds
    max_observation_time : float
        maximum observation time in seconds
    """
    amplitudes, lifetimes = np.reshape(params, (2, -1))
    components = exponential_mixture_log_likelihood_components(
        amplitudes, lifetimes, t, min_observation_time, max_observation_time
    )
    log_likelihood = logsumexp(components, axis=0)
    return -np.sum(log_likelihood)


def _handle_amplitude_constraint(
    n_components, params, fixed_param_mask
) -> Tuple[np.ndarray, Union[Dict, Tuple], np.ndarray]:
    """Determines how many amplitudes actually need fitting.

    For a single-component model the amplitude is 1 by definition. For an N-component model, where
    N - 1 amplitudes are fixed, the free amplitude is determined by the constraint. For models that
    have more than one free amplitude, the amplitudes are constrained to sum to 1.

    Parameters
    ----------
    n_components : int
        number of components in the mixture model
    params : array_like
        model parameters
    fixed_param_mask : array_like
        logical mask of fixed parameters

    Returns
    -------
    Tuple[np.ndarray, Union[Dict, Tuple], np.ndarray]
        This function returns the mask of parameters to be fitted, amplitude constraint function
        and updated parameter vector (forced to be consistent with the constraint).

    Raises
    ------
    ValueError
        If the sum of the provided fixed amplitudes exceeds 1.
    ValueError
        If all amplitudes are fixed but the amplitudes do not sum to 1.
    """
    if fixed_param_mask is None:
        fixed_param_mask = np.zeros(params.shape, dtype=bool)
    elif len(fixed_param_mask) != len(params):
        raise ValueError(
            f"Length of fixed parameter mask ({len(fixed_param_mask)}) is not equal to the number "
            f"of model parameters ({len(params)})"
        )

    is_amplitude = np.hstack(
        (np.ones(n_components, dtype=bool), np.zeros(n_components, dtype=bool))
    )

    # Contribution from amplitudes that are fixed.
    fixed_amplitude_mask = np.logical_and(is_amplitude, fixed_param_mask)
    if (sum_fixed_amplitudes := np.sum(params[fixed_amplitude_mask])) > 1:
        raise ValueError(
            f"Invalid model. Sum of the fixed amplitudes is bigger than 1 ({sum_fixed_amplitudes})."
        )

    # Determine what actually needs to be fitted
    fitted_param_mask = np.logical_not(fixed_param_mask)
    free_amplitudes = np.logical_and(is_amplitude, fitted_param_mask)
    num_free_amps = np.sum(free_amplitudes)

    # If we are only fitting a single amplitude at this point (i.e. 1-component model, 2-component
    # model with 1 amplitude fixed, or N_component model with N-1 components fixed), we can simply
    # set it to its correct value and fix it (since there is only 1 degree of freedom).
    if num_free_amps == 1:
        free_amplitude_idx = np.nonzero(free_amplitudes)[0]
        fitted_param_mask[free_amplitude_idx] = False
        params = params.copy()  # Make sure we don't modify the input variable
        params[free_amplitude_idx] = 1.0 - sum_fixed_amplitudes
        sum_fixed_amplitudes += params[free_amplitude_idx]
        num_free_amps -= 1

    if num_free_amps == 0 and not np.allclose(sum_fixed_amplitudes, 1.0, atol=1e-6):
        raise ValueError(
            f"Invalid model. Sum of the provided amplitudes has to be 1 ({sum_fixed_amplitudes})."
        )

    constraints = (
        {
            "type": "eq",
            "fun": lambda x, n: 1 - sum(x[:n]) - sum_fixed_amplitudes,
            "args": [num_free_amps],
        }
        if num_free_amps > 0
        else ()
    )

    return fitted_param_mask, constraints, params


def _exponential_mle_bounds(n_components, min_observation_time, max_observation_time):
    return (
        *[(1e-9, 1.0) for _ in range(n_components)],
        *[(min_observation_time * 0.1, max_observation_time * 1.1) for _ in range(n_components)],
    )


def _exponential_mle_optimize(
    n_components,
    t,
    min_observation_time,
    max_observation_time,
    initial_guess=None,
    options=None,
    fixed_param_mask=None,
):
    """Calculate the maximum likelihood estimate of the model parameters given measured dwelltimes.

    Parameters
    ----------
    n_components : int
        number of components in the mixture model
    t : array_like
        dwelltime observations in seconds
    min_observation_time : float
        minimum observation time in seconds
    max_observation_time : float
        maximum observation time in seconds
    initial_guess : array_like, optional
        initial guess for the model parameters ordered as
        [amplitude1, amplitude2, ..., lifetime1, lifetime2, ...]
    options : dict, optional
        additional optimization parameters passed to `minimize(..., options)`.
    fixed_param_mask : array_like, optional
        logical mask of which parameters to fix during optimization. When omitted, no parameter is
        assumed fixed.

    Raises
    ------
    ValueError
        If the sum of the provided fixed amplitudes in the initial_guess exceeds 1.
    ValueError
        If all amplitudes are fixed but the amplitudes in the initial_guess do not sum to 1.
    """
    if np.any(np.logical_or(t < min_observation_time, t > max_observation_time)):
        raise ValueError(
            "some data is outside of the bounded region. Please choose"
            "appropriate values for `min_observation_time` and/or `max_observation_time`."
        )

    if initial_guess is None:
        initial_guess_amplitudes = np.ones(n_components) / n_components
        initial_guess_lifetimes = np.mean(t) * np.arange(1, n_components + 1)
        initial_guess = np.hstack([initial_guess_amplitudes, initial_guess_lifetimes])

    bounds = _exponential_mle_bounds(n_components, min_observation_time, max_observation_time)

    fitted_param_mask, constraints, initial_guess = _handle_amplitude_constraint(
        n_components, initial_guess, fixed_param_mask
    )

    current_params = initial_guess.copy()

    def cost_fun(params):
        current_params[fitted_param_mask] = params

        return exponential_mixture_log_likelihood(
            current_params,
            t=t,
            min_observation_time=min_observation_time,
            max_observation_time=max_observation_time,
        )

    # Nothing to fit, return!
    if np.sum(fitted_param_mask) == 0:
        return initial_guess, -cost_fun([])

    result = minimize(
        cost_fun,
        current_params[fitted_param_mask],
        method="SLSQP",
        constraints=constraints,
        bounds=[bound for bound, fitted in zip(bounds, fitted_param_mask) if fitted],
        options=options,
    )

    # output parameters as [amplitudes, lifetimes], -log_likelihood
    current_params[fitted_param_mask] = result.x
    return current_params, -result.fun


def _dwellcounts_from_statepath(statepath, exclude_ambiguous_dwells):
    """Calculate the dwell counts and slicing indices for all states in a state path trajectory.

    Note: the counts are the number of frames or time points. To convert to proper
    dwelltimes, multiply the counts by the time step between points.

    Parameters
    ----------
    statepath : array_like
        Time-ordered array of state labels
    exclude_ambiguous_dwells : bool
        Determines whether to exclude dwelltimes which are not exactly determined. If `True`, the first
        and last dwells are not used in the analysis, since the exact start/stop times of these events are
        not definitively known.

    Returns
    -------
    dict:
        Dictionary of all dwell counts for each state. Keys are state labels.
    dict:
        Dictionary of slicing indices for all dwells for each state. Keys are state labels.
    """
    unique_states = np.unique(statepath)
    # pad with extra state to catch first and last dwells
    # this also effectively shifts our indexing by one so we
    # keep the exclusive last index
    assert np.all(np.isfinite(statepath))
    padded_statepath = np.hstack((np.nan, statepath, np.nan))

    # store list slicing indices for each dwell in a state
    state_ranges = {}
    for state in unique_states:
        mask = np.array(padded_statepath == state).astype(int)
        diff_mask = np.diff(mask)

        # find slicing indices as [start:stop) pairs
        # as rows in array
        idx = np.argwhere(diff_mask != 0).squeeze()
        idx = idx.reshape((-1, 2))

        if exclude_ambiguous_dwells:
            start = 1 if idx[0, 0] == 0 else 0
            stop = -1 if idx[-1, 1] == len(statepath) else None
            idx = idx[start:stop]

        state_ranges[state] = idx

    dwell_counts = {key: np.diff(idx, axis=1).squeeze() for key, idx in state_ranges.items()}

    return dwell_counts, state_ranges
