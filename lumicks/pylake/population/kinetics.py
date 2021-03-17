import numpy as np
from scipy import stats
from .detail.random import draw_bootstrap_indices


def exponential_mle(t, t_min=0):
    """Maximum likelihood estimate of scale parameter (tau)
    corrected for minimum observation time.
    Return rate parameter (k = 1/tau)."""
    tau = np.mean(t) - t_min
    return 1 / tau


class ExponentialDistribution:
    def __init__(self, data, t_min=0, bootstrap_iter=500):
        """An exponential probability distribution with parameters estimated
        from a data sample by maximum likelihood estimation.

        Parameters
        ----------
        data : array
            observed data sample
        t_min : float
            minimum data acquisition time
        bootstrap_iter : int
            number of bootstrapping iterations used to estimate parameter mean
        """

        self._data = data
        self._n = len(data)

        mle_results = [
            exponential_mle(data[draw_bootstrap_indices(self._n)], t_min=t_min)
            for _ in range(bootstrap_iter)
        ]
        self._rate = np.mean(mle_results)
        self._sem = stats.sem(mle_results)

    @property
    def rate(self):
        """Rate parameter."""
        return self._rate

    def ci(self, alpha=0.95):
        """Calculate confidence interval at alpha level."""
        return stats.t.interval(alpha, self._n - 1, loc=self.rate, scale=self._sem)

    def pdf(self, t):
        """Probability distribution function."""
        return self.rate * np.exp(-self.rate * t)

    def hist(self, n_bins=50, plot_kwargs={}, hist_kwargs={}):
        """Plot a histogram of the data overlaid with the model PDF.
        Parameters
        ----------
        n_bins : int
            number of histogram bins
        plot_kwargs : dict
            plotting keyword arguments passed to the PDF line plot
        hist_kwargs : dict
            plotting keyword arguments passed to the histogram plot
        """
        import matplotlib.pyplot as plt

        hist_kwargs = {"fc": "#c5c5c5", "ec": "#b5b5b5", **hist_kwargs}
        plot_kwargs = {"lw": 2, **plot_kwargs}

        bins, step = np.linspace(np.min(self._data), np.max(self._data), n_bins, retstep=True)
        t = np.linspace(bins[0] + step / 2, bins[-1] - step / 2, 300)
        plt.hist(self._data, bins=bins, density=True, **hist_kwargs)
        plt.plot(t, self.pdf(t), **plot_kwargs)
