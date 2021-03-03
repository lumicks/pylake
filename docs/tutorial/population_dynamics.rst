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
