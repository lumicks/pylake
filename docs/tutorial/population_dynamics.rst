.. warning::
    This is early access alpha functionality. While usable, this has not yet been tested in a large number of different
    scenarios. The API is still be subject to change *without any prior deprecation notice*! If you use this
    functionality keep a close eye on the changelog for any changes that may affect your analysis.

Population Dynamics
===================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`


The following tools enable analysis of experiments which can be described as a system of discrete states. Consider as an example
a DNA hairpin which can exist in a folded and unfolded state. At equilibrium, the relative populations of the two states are
governed by the interconversion kinetics described by the rate constants \\(k_{fold}\\) and \\(k_{unfold}\\)

.. image:: hairpin_kinetics.png

First let's take a look at some example force channel data. We'll downsample the data in order to speed up the calculations
and smooth some of the noise::

    raw_force = file.force1x
    force = raw_force.downsampled_by(78)

    raw_force.plot()
    force.plot(start=raw_force.start)
    plt.xlim(0, 3)

.. image:: pop_hairpin_trace.png

Care must be taken in downsampling the data so as not to introduce signal artifacts in the data (see below for a more detailed discussion).

The goal of the analyses described below is to label each data point in a state in order to extract
kinetic and thermodynamic information about the underlying system.

Models
^^^^^^

Gaussian Mixture Models
-----------------------

The Gaussian Mixture Model (GMM) is a simple probabilistic model that assumes all data points belong
to one of a fixed number of states, each of which is described by a (weighted) normal distribution.

We can initialize a model from a force channel as follows::

    gmm = lk.GaussianMixtureModel.from_channel(force, n_states=2)
    print(gmm.weights) # [0.53070846 0.46929154]
    print(gmm.means)   # [10.72850757 12.22295543]
    print(gmm.std)     # [0.21461742 0.21564235]

Here the force data is used to train a 2 state GMM. The weights give the fraction of time spent each state
while the means and standard deviations indicate the average signal and noise for each state, respectively.

We can visually inspect the quality of the fitted model by plotting a histogram of the data overlaid with the weighted normal distribution probability density functions::

    gmm.hist(force)

.. image:: pop_gmm_hist.png

We can also plot the time trace with each data point labeled with the most likely state::

    gmm.plot(force['0s':'2s'])

.. image:: pop_gmm_labeled_trace.png

Note that the data does not have to be the same data used to train the model.

Data Pre-processing
^^^^^^^^^^^^^^^^^^^

It can be desirable to downsample the raw channel data in order to decrease the number of data points used
by the model training algorithm (in order to speed up the calculation) and to smooth experimental noise.
However, great care must be taken in doing so in order to avoid introducing artifacts into the signal.

As shown in the histograms below, as the data is downsampled the state peaks narrow considerably, but density
between the peaks remains (indicated by the arrow). These intermediate data points are the result of averaging over a span of data from
two different states. The result is the creation of a new state that does not arise from any physically relevant mechanism.

.. image:: pop_downsampling_notes.png

Furthermore, states with very short lifetimes can be averaged out of the data if the downsampling factor is too high. Therefore,
in order to ensure robust results, it may be advisable to carry out the analysis at a few different downsampled rates.

Dwelltime analysis
------------------

The lifetimes of bound states can be estimated by fitting to an Exponential distribution. The `DwelltimeModel` class can be used
to optimize the model parameters for an array of determined dwelltimes::

    dwell_1 = lk.DwelltimeModel(dwelltimes_seconds, n_components=1)

The model is optimized using Maximum Likelihood Estimation (MLE) :cite:`kaur2019dwell,woody2016memlet`. The advantage of this method
is that it does not require binning the data. The number of exponential components to be used for the fit is chosen with the `n_components` argument.

We can visually inspect the result with::

    dwell_1.plot(bin_spacing="log")

.. image:: kymo_bind_dwell_1.png

The `bin_spacing` argument can be either `"log"` or `"linear"` and controls the spacing of the bin edges.
The scale of the x- and y-axes can be controlled with the optional `xscale` and `yscale` arguments; if they are not specified
the default visualization is `lin-lin` for `bin_spacing="linear"` and `lin-log` for `bin_spacing="log"`.

You can optionally pass the number of bins to be plotted as `n_bins`. Note that the number of bins
is purely for visualization purposes; the model is optimized directly on the unbinned dwelltimes. This is the main
advantage of the MLE method over analyses that use a least squares fitting to binned data, where the bin widths and number
of bins can drastically affect the optimized parameters.

We can clearly see that this distribution is not fit well by a single exponential decay.
Let's now see what a double exponential distribution looks like::

    dwell_2 = traces.fit_binding_times(n_components=2)
    dwell_2.plot(bin_spacing="log")

.. image:: kymo_bind_dwell_2.png

Here we see that the double exponential fit visually looks better and the log likelihood is also higher than that
for the single exponential fit. However, the log likelihood does not take into account model complexity, and will
always increase for a model with more degrees of freedom. Instead, we can look at the Bayesian Information Criterion (BIC)
or Akaike Information Criterion (AIC) to determine which model is better::

    >>> print(dwell_1.bic, dwell_1.aic)
    532.3299315589168  529.0366267341923

    >>> print(dwell_2.bic, dwell_2.aic)
    520.4562630650156  510.5763485908421

These information criterion values weigh the log likelihood against the model complexity, and as such are more useful for
model selection. In general, the model with the lowest value is optimal. We can see that both values are lower for the double
exponential model, indicating that it is a better fit to the data.

We can see this effect if we purposely overfit the data. The following plot shows the result of fitting simulated data (randomly sampled
from a single exponential distribution) with either a one- or two-component model. In the figure legends we see that the log likelihood
increases slightly for the two-component model because of the larger degrees of freedom. However, the BIC for the one-component model is
indeed lower, as expected:

.. image:: kymo_bic_compare.png

Going back to our experimental data, we can next attempt to estimate confidence intervals (CI) for the parameters using bootstrapping.
Here, a random dataset with the same size as the original is sampled (with replacement) from the original dataset. This sampled dataset
is then fit using the MLE method, just as for the original dataset. The fit results in a new estimate for the model parameters.
This process is repeated many times, and the distribution of the resulting parameters can be analyzed to estimate certain statistics about the them::

    dwell_2.calculate_bootstrap(iterations=1000)
    dwell_2.bootstrap.plot(alpha=0.05)

.. image:: kymo_bind_bootstrap_2.png

Here we see the distributions of the bootstrapped parameters. The vertical lines indicate the
means of the distributions, while the red area indicates the estimated confidence intervals. The `alpha` argument determines
the CI that is estimated as `100*(1-alpha)` % CI; in this case we're showing the estimate for the 95% CI. The values for the
lower and upper bounds are the `100*(alpha/2)` and `100*(1-alpha/2)` percentiles of the distributions.

Note, however, that while the means correspond well with the optimized model parameters, the distributions are not symmetric.
In such a case, the simple method of using percentiles as CI values may not be appropriate. For more advanced analysis,
the distribution values are directly available through the properties `DwelltimeModel.bootstrap.amplitude_distributions` and
`DwelltimeModel.bootstrap.lifetime_distributions` which return the data as a `numpy` array with
shape `[# components, # bootstrap samples]`.

The Exponential (Mixture) Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model likelihood :math:`\mathcal{L}` is defined for a mixture of exponential distributions as:

.. math::

    \mathcal{L} = \prod_j^T \left[ \frac{1}{N} \sum_i^M \frac{a_i}{\tau_i} \exp{\left( \frac{-t_j}{\tau_i} \right)} \right]

where :math:`T` is the number of observed dwell times, :math:`M` is the number of exponential components, :math:`t` is time,
:math:`\tau_i` is the lifetime of component :math:`i`, and :math:`a_i` is the fractional contribution of component :math:`i`
under the constraint of :math:`\sum_i^M a_i = 1`. The normalization constant :math:`N` is defined as:

.. math::

    N = \sum_i^M a_i \left[
    \exp{ \left( \frac{-t_{min}}{\tau_i} \right)} -
    \exp{ \left( \frac{-t_{max}}{\tau_i} \right)}
    \right]

where :math:`t_{min}` and :math:`t_{max}` are the minimum and maximum possible observation times.

The normalization constant takes into account the minimum and maximum possible observation times of the experiment. These
can be set manually with the `min_observation_time` and `max_observation_time` keyword arguments, respectively. The default
values are :math:`t_{min}=0` and :math:`t_{max}=\infty`, such that :math:`N=1`. However, for real experimental data,
there are physical limitations on the measurement times (such as pixel integration time for kymographs or sampling frequency for
force channels) that should be taken into account.
