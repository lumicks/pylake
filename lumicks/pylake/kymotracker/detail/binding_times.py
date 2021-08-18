import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from functools import partial
from dataclasses import dataclass, field
import matplotlib.pyplot as plt


@dataclass
class BindingDwelltimesBootstrap:
    """Bootstrap distributions for a binding dwelltime model.

    This class is stored in the `BindingDwelltime.bootstrap` attribute
    and should not be constructed manually.

    Attributes
    ----------
    _samples : np.ndarray
        array of optimized model parameters for each bootstrap sample pull; shape is
        [number of parameters, number of samples]
    """

    _samples: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    def _sample_distributions(self, optimized, iterations):
        """Construct bootstrap distributions for parameters.

        For each iteration, a dataset is randomly selected (with replacement) with the same
        size as the data used to optimize the model. Model parameters are then optimized
        for this new sampled dataset.

        Parameters
        ----------
        optimized : BindingDwelltimes
            optimized model results
        iterations : int
            number of iterations (random samples) to use for the bootstrap
        """
        n_data = optimized.dwelltimes_sec.size
        self._samples = np.empty((optimized._parameters.size, iterations))
        for itr in range(iterations):
            sample = np.random.choice(optimized.dwelltimes_sec, size=n_data, replace=True)
            result = _kinetic_mle_optimize(
                optimized.n_components,
                sample,
                *optimized.observation_limits,
                initial_guess=optimized._parameters,
            )
            self._samples[:, itr] = result._parameters

    @property
    def n_samples(self):
        """Number of samples in the bootstrap."""
        return self._samples.shape[1]

    @property
    def n_components(self):
        """Number of components in the model."""
        return int(self._samples.shape[0] / 2)

    @property
    def amplitude_distributions(self):
        """Array of sample optimized amplitude parameters; shape is
        [number of components, number of samples]"""
        return self._samples[: self.n_components]

    @property
    def lifetime_distributions(self):
        """Array of sample optimized lifetime parameters; shape is
        [number of components, number of samples]"""
        return self._samples[self.n_components :]

    def calculate_stats(self, key, component, alpha=0.05):
        """Calculate the mean and confidence intervals of the bootstrap distribution for a parameter.

        *NOTE*: the `100*(1-alpha)` % confidence intervals calculated here correspond to the
        `100*(alpha/2)` and `100*(1-(alpha/2))` quantiles of the distribution. For distributions
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
        mean = np.mean(data)
        lower = np.quantile(data, alpha / 2)
        upper = np.quantile(data, 1 - (alpha / 2))
        return mean, (lower, upper)

    def plot(self, alpha=0.05, n_bins=25, hist_kwargs={}, span_kwargs={}, line_kwargs={}):
        """Plot the bootstrap distributions for the parameters of a model.

        Parameters
        ----------
        alpha : float
            confidence intervals are calculated as 100*(1-alpha)%
        n_bins : int
            number of bins in the histogram
        hist_kwargs : dict
            dictionary of plotting kwargs applied to histogram
        span_kwargs : dict
            dictionary of plotting kwargs applied to the patch indicating the area
            spanned by the confidence intervals
        line_kwargs : dict
            dictionary of plotting kwargs applied to the line indicating the
            distribution means
        """
        hist_kwargs = {"facecolor": "#c5c5c5", "edgecolor": "#888888", **hist_kwargs}
        span_kwargs = {"facecolor": "tab:red", "alpha": 0.3, **span_kwargs}
        line_kwargs = {"color": "k", **line_kwargs}

        def plot_axes(data, key, component, use_index):
            plt.hist(data, bins=n_bins, **hist_kwargs)
            mean, (lower, upper) = self.calculate_stats(key, component, alpha)
            plt.axvspan(lower, upper, **span_kwargs)
            plt.axvline(mean, **line_kwargs)
            plt.xlabel(f"{key}" if key == "amplitude" else f"{key} (sec)")
            plt.ylabel("counts")

            label = "a" if key == "amplitude" else r"\tau"
            unit = "" if key == "amplitude" else "sec"
            prefix = fr"${label}_{component+1}$" if use_index else fr"${label}$"
            plt.title(f"{prefix} = {mean:0.2g} ({lower:0.2g}, {upper:0.2g}) {unit}")

        if self.n_components == 1:
            data = self.lifetime_distributions.squeeze()
            plot_axes(data, "lifetime", 0, False)
        else:
            for component in range(2):
                for column, key in enumerate(("amplitude", "lifetime")):
                    data = getattr(self, f"{key}_distributions")[component]
                    column += 1
                    plt.subplot(self.n_components, 2, 2 * component + column)
                    plot_axes(data, key, component, True)

        plt.tight_layout()


@dataclass(frozen=True)
class BindingDwelltimes:
    """Results of exponential mixture model optimization for binding dwelltimes.

    This class is returned from `_kinetic_mle_optimize()` and should not be
    constructed manually.

    Attributes
    ----------
    n_components : int
        number of components in the model.
    dwelltimes_sec : np.ndarray
        observations on which the model was trained.
    observations_limits : tuple
        tuple of (`min`, `max`) values of the experimental observation time.
    _parameters : np.ndarray
        optimized parameters in the order [amplitudes, lifetimes]
    log_likelihood : float
        log likelihood of the trained model
    bootstrap : BindingDwelltimesBootstrap
        object containing information about the bootstrapping analysis.
    """

    n_components: int
    dwelltimes_sec: np.ndarray = field(repr=False)
    observation_limits: list = field(repr=False)
    _parameters: np.ndarray = field(repr=False)
    log_likelihood: float
    bootstrap: BindingDwelltimesBootstrap = field(
        default_factory=BindingDwelltimesBootstrap, init=False, repr=False
    )

    @property
    def amplitudes(self):
        """Fractional amplitude of each model component"""
        return self._parameters[: self.n_components]

    @property
    def lifetimes(self):
        """Lifetime parameter (in seconds) of each model component."""
        return self._parameters[self.n_components :]

    @property
    def aic(self):
        """Akaike Information Criterion."""
        k = (2 * self.n_components) - 1  # number of parameters
        return 2 * k - 2 * self.log_likelihood

    @property
    def bic(self):
        """Bayesian Information Criterion."""
        k = (2 * self.n_components) - 1  # number of parameters
        n = self.dwelltimes_sec.size  # number of observations
        return k * np.log(n) - 2 * self.log_likelihood

    def calculate_bootstrap(self, iterations=500):
        self.bootstrap._sample_distributions(self, iterations)

    def plot(
        self,
        n_bins=25,
        bin_spacing="linear",
        hist_kwargs={},
        component_kwargs={},
        fit_kwargs={},
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
        hist_kwargs : dict
            dictionary of plotting kwargs applied to histogram
        component_kwargs : dict
            dictionary of plotting kwargs applied to the line plot for each component
        fit_kwargs : dict
            dictionary of plotting kwargs applied to line plot for the total fit
        xscale : {"log", "linear", None}
            scaling for the x-axis; when `None` default is "linear"
        yscale : {"log", "linear", None}
            scaling for the y-axis; when `None` default is same as `bin_spacing`
        """
        if bin_spacing == "log":
            scale = np.logspace
            limits = (np.log10(self.dwelltimes_sec.min()), np.log10(self.dwelltimes_sec.max()))
            xscale = "linear" if xscale is None else xscale
            yscale = "log" if yscale is None else yscale
        elif bin_spacing == "linear":
            scale = np.linspace
            limits = (self.dwelltimes_sec.min(), self.dwelltimes_sec.max())
            xscale = "linear" if xscale is None else xscale
            yscale = "linear" if yscale is None else yscale
        else:
            raise ValueError("spacing must be either 'log' or 'linear'")

        bins = scale(*limits, n_bins)
        centers = bins[:-1] + (bins[1:] - bins[:-1]) / 2

        hist_kwargs = {"facecolor": "#cdcdcd", "edgecolor": "#aaaaaa", **hist_kwargs}
        component_kwargs = {"marker": "o", "ms": 3, **component_kwargs}
        fit_kwargs = {"color": "k", **fit_kwargs}

        components = np.exp(
            exponential_mixture_log_likelihood_components(
                self.amplitudes, self.lifetimes, centers, *self.observation_limits
            )
        )

        def label_maker(a, t, n):
            if self.n_components == 1:
                amplitude = ""
                lifetime_label = r"$\tau$"
            else:
                amplitude = f"($a_{n}$ = {a:0.2g}) "
                lifetime_label = fr"$\tau_{n}$"
            return f"{amplitude}{lifetime_label} = {t:0.2g} sec"

        # plot histogram
        density, _, _ = plt.hist(self.dwelltimes_sec, bins=bins, density=True, **hist_kwargs)
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
    params = np.reshape(params, (2, -1))
    components = exponential_mixture_log_likelihood_components(
        params[0], params[1], t, min_observation_time, max_observation_time
    )
    log_likelihood = logsumexp(components, axis=0)
    return -np.sum(log_likelihood)


def _kinetic_mle_optimize(
    n_components, t, min_observation_time, max_observation_time, initial_guess=None
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
    initial_guess : array_like
        initial guess for the model parameters ordered as
        [amplitude1, amplitude2, ..., lifetime1, lifetime2, ...]
    """
    if np.any(np.logical_or(t < min_observation_time, t > max_observation_time)):
        raise ValueError(
            "some data is outside of the bounded region. Please choose"
            "appropriate values for `min_observation_time` and/or `max_observation_time`."
        )

    cost_fun = partial(
        exponential_mixture_log_likelihood,
        t=t,
        min_observation_time=min_observation_time,
        max_observation_time=max_observation_time,
    )

    if initial_guess is None:
        initial_guess_amplitudes = np.ones(n_components) / n_components
        initial_guess_lifetimes = np.mean(t) * np.arange(1, n_components + 1)
        initial_guess = np.hstack([initial_guess_amplitudes, initial_guess_lifetimes])

    bounds = (
        *[(np.finfo(float).eps, 1) for _ in range(n_components)],
        *[(min_observation_time * 0.1, max_observation_time * 1.1) for _ in range(n_components)],
    )
    constraints = {"type": "eq", "fun": lambda x, n: 1 - sum(x[:n]), "args": [n_components]}
    result = minimize(
        cost_fun, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints
    )

    return BindingDwelltimes(
        n_components, t, (min_observation_time, max_observation_time), result.x, -result.fun
    )
