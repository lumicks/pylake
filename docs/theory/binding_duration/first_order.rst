Modeling first order kinetics
=============================

Classic ensemble experiments
----------------------------

Let's revist the first order reaction of a protein dissociating from a DNA substrate. The rate
is described by:

.. math::

    -\frac{\mathrm{d}[DP]}{\mathrm{d}t} = k_\mathrm{off}[DP]

Since we're intersted in obtaining the value of the rate constant :math:`k_\mathrm{off}`, we first
rearrange this equation slightly and integrate both sides:

.. math::

    \int_{[DP]_0}^{[DP]} \frac{1}{[DP]} \: \mathrm{d}[DP]
    = - \int_{t_0}^t k_\mathrm{off} \: \mathrm{d}t

where :math:`[DP]_0` and :math:`t_0` indicate the initial concentration of the complex and
initial time, respectively.

After carrying out the intergration, rearranging slightly and raising each side by the natural
exponent, we arrive at the standard integrated form of the first order rate law:

.. math::

    [DP] = [DP]_0 e^{-k_\mathrm{off}t}

Here we see that the concentration of the complex decays exponentially from the initial
concentration. Indeed, the standard method for determing the first order rate constant in
bulk experiments is to monitor the concentration of the reactant over time and fit the resulting
curve to this equation.

.. figure:: figures/sim_bulk_decay.png

Single molecule experiments
---------------------------

From the single molecule perspective, the approach is slightly different. Because we can monitor
individual complexes, thinking in terms of concentration is no longer useful. Instead, we turn
to a stochastic formulation of the kinetics of the reaction. Given that we start with a single bound
complex, the question of kinetics now becomes "when will the dissociation reaction occur?" The time
spent in one state before transitioning to a new state is often referred to as the *dwell time*

Chemical reactions at the molecular scale can be described as a Markov process: a
random process where the transition from one state to another is determined only by the present
state of the system and the probability of transitioning between the two states.
Mathematically, the probability for a reaction to occur at a time :math:`t` is described by an
exponential distribution, with a probability distribution function (PDF) of the form:

.. math::

    p(t | k_\mathrm{off}) = k_\mathrm{off} e^{-k_\mathrm{off}t}

In single molecule experiments, often we can observe the transition between two states and therefore
measure the dwell time for a particular reaction. Combining observation for many transitions, it is
then possible to model this experimental distribution of dwell times with the above PDF to obtain
an estimate for the stochastic rate constant.

Let's go back to our example of a protein dissociating from a DNA tether. Experimentally, we can
easily measure the binding dwell time from tracked particles on a kymograph:

.. figure:: figures/kymo_zoom_dwells.png

If we combine the measured dwell times from many events we can then fit the distribution with the
equation above to obain the rate constant:

.. figure:: figures/sim_dwell_hist.png
