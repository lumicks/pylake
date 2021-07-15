.. warning::
    This is beta functionality. While usable, this has not yet been tested in a large
    number of different scenarios. The API may also still be subject to change.

Fd Fitting
==========

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

Pylake is capable of fitting force extension curves using various polymer models. This tutorial aims at being a gentle
introduction to these fitting capabilities.

Models
------

When fitting data, everything revolves around models. One of these models is the so-called extensible worm-like-chain
model by Odijk et al. Let's have a look at it. We can construct this model using the supplied function `lk.odijk()`.

Note that we also have to give it a name. This name will be prefixed to model specific parameters in this model::

    >>> model = lk.odijk("DNA")

Entering model prints the model equation and its default parameters::

    >>> model

    Model equation:

    d(f) = DNA.Lc * (1 - (1/2)*sqrt(kT/(f*DNA.Lp)) + f/DNA.St)

    Parameter defaults:

    Name      Value  Unit      Fitted      Lower bound    Upper bound
    ------  -------  --------  --------  -------------  -------------
    DNA/Lp    40     [nm]      True                  0            inf
    DNA/Lc    16     [micron]  True                  0            inf
    DNA/St  1500     [pN]      True                  0            inf
    kT         4.11  [pN*nm]   False                 0              8

Note how most of the parameters listed have the model name prefixed. For example, the persistence length in this model
is now named `DNA/Lp`. Here `DNA` refers to the name we gave to the model.

One parameter stands out, which is kT as it doesn't have `DNA` prefixed. The reason for this is that this parameter is
not model specific (in this case the parameter represents the Boltzmann constant times the temperature).

We can also obtain the parameters as a list::

    >>> print(model.parameter_names)

    ['DNA/Lp', 'DNA/Lc', 'DNA/St', 'kT']

As we can see from the model equation, the model is given as distance as a function of force.

Simulating the model
--------------------

We can simulate the model by passing a dictionary with parameters values::

    dna = lk.odijk("DNA")
    force = np.arange(0.1, 14, 0.1)
    dna(force, {"DNA/Lp": 50.0, "DNA/Lc": 16.0, "DNA/St": 1500.0, "kT": 4.11})

Note how the model name is prefixed to the model specific parameters, forming a key-value pair like so: `"name/parameter": value`.

Model composition and inversion
-------------------------------

In practice, we would typically want to fit force as a function of distance however. For this we have the inverted
Odijk model::

    model = lk.inverted_odijk("DNA")

We can have a quick look at what this model looks like and which parameters are in there::

    >>> model

    Model equation:

    f(d) = argmin[f](norm(DNA.Lc * (1 - (1/2)*sqrt(kT/(f*DNA.Lp)) + f/DNA.St)-(d - DNA.d_offset))) + DNA.f_offset

    Parameter defaults:

    Name      Value  Unit      Fitted      Lower bound    Upper bound
    ------  -------  --------  --------  -------------  -------------
    DNA/Lp    40     [nm]      True                  0            inf
    DNA/Lc    16     [micron]  True                  0            inf
    DNA/St  1500     [pN]      True                  0            inf
    kT         4.11  [pN*nm]   False                 0              8

This seems to be what we want: force as a function of distance. Let's assume we have acquired data, but upon analysis
we notice that the template matching wasn't completely optimal. In addition, we experienced some force drift, while
forgetting to reset the force back to zero. In this case, we can incorporate offsets in our model. We can introduce an
offset in the independent parameter, by calling `.subtract_independent_offset()` on our model::

    >>> model = lk.inverted_odijk("DNA").subtract_independent_offset()
    >>> model

    Model: DNA(x-d)

    Model equation:

    f(d) = argmin[f](norm(DNA.Lc * (1 - (1/2)*sqrt(kT/(f*DNA.Lp)) + f/DNA.St)-(d - DNA.d_offset)))

    Parameter defaults:

    Name            Value  Unit      Fitted      Lower bound    Upper bound
    ------------  -------  --------  --------  -------------  -------------
    DNA/d_offset     0.01  [au]      True               -0.1            0.1
    DNA/Lp          40     [nm]      True                0            100
    DNA/Lc          16     [micron]  True                0            inf
    DNA/St        1500     [pN]      True                0            inf
    kT               4.11  [pN*nm]   False               0              8

If we also expect an offset in the dependent parameter, we can simply add an offset model to our model::

    >>> model = lk.inverted_odijk("DNA").subtract_independent_offset() + lk.force_offset("DNA")
    >>> model

    Model: DNA(x-d)_with_DNA

    Model equation:

    f(d) = argmin[f](norm(DNA.Lc * (1 - (1/2)*sqrt(kT/(f*DNA.Lp)) + f/DNA.St)-(d - DNA.d_offset))) + DNA.f_offset

    Parameter defaults:

    Name            Value  Unit      Fitted      Lower bound    Upper bound
    ------------  -------  --------  --------  -------------  -------------
    DNA/d_offset     0.01  [au]      True               -0.1            0.1
    DNA/Lp          40     [nm]      True                0            100
    DNA/Lc          16     [micron]  True                0            inf
    DNA/St        1500     [pN]      True                0            inf
    kT               4.11  [pN*nm]   False               0              8
    DNA/f_offset     0.01  [pN]      True               -0.1            0.1

From the above example, you can see how easy it is to composite models. Sometimes, models become more complicated. For
instance, we may have two worm like chain models that we wish to add, and then invert. For the Odijk model, this can be
done as follows::

    model = lk.odijk("DNA") + lk.odijk("protein") + lk.distance_offset("offset")
    model = model.invert()

Note how we added three models and then inverted the composition of those models. Models inverted via `invert()` will
typically be slower than the pre-inverted counterparts. This is because the inversion is done numerically rather than
analytically. For more complex examples on how this inversion may be used, please see the examples.

For a full list of models that are available, please refer to the documentation by invoking `help(lk.fitting.models)`
or see :ref:`fd_models`.

Fitting data
------------

To fit Fd models, we have to create an `FdFit`. This object will collect all the parameters involved in the models and
data, and will allow you to interact with the model parameters and fit them. We construct it using `lk.FdFit` and
pass it one or more models. In return, we get an object we can interact with, which in this case we store in `fit`::

    fit = lk.FdFit(model)

Adding data to the fit
**********************

To do a fit, we have to add data. Let's assume we have two data sets. One was acquired in the presence of a ligand, and
another was measured without a ligand. We expect this ligand to only affect the contour length of our DNA. Let's add the
first data set which we name `Control`. Adding it to the fit is simple::

    fit.add_data("Control", force1, distance1)

For the second data set, we want the contour length to be different. We can achieve this by renaming the parameter
when loading the data::

    fit.add_data("RecA", force2, distance2, params={"DNA/Lc": "DNA/Lc_RecA"})

More specifically, we renamed the parameter `DNA/Lc` to `DNA/Lc_RecA`.

Setting parameter bounds
************************

The parameters of the model can be accessed directly from `FdFit`. Note that by default, parameters tend to have
reasonable initial guesses and bounds in pylake, but we can set our own as follows::

    fit["DNA/Lp"].value = 50
    fit["DNA/Lp"].lower_bound = 39
    fit["DNA/Lp"].upper_bound = 80

After this, the model is ready to be fitted. We can fit the model to the data by calling the function `.fit()`. This
estimates the model parameters by minimizing the least squares differences between the model's dependent variable and
the data in the fit::

    fit.fit()

After this call, the parameters will have new values that should bring the model closer to the data. Note that multiple
models can be fit at once by supplying more than one model::

    fit = lk.FdFit(model1, model2, model3)

Frequently, global fits have better statistical properties than fitting the data separately as more information is
available to infer parameters shared between the various models.


Plotting the data
-----------------

A model can be plotted before it is fitted. This can be useful when the default parameter values don't seem to work
very well. Parameter estimation is typically initiated from an initial guess. A poor initial guess can lead to a poor
parameter estimate. Therefore, you might want to see what your initial model curve looks like and set some better
initial guesses yourself when you run into trouble.


Fits can be plotted using the built-in plot functionality::
    
    fit.plot()
    plt.ylabel("Force [pN]")
    plt.xlabel("Distance [$\\mu$M]");

Sometimes, more fine grained control over the plots is required. Let's say we want to plot the model over a range of
values (in this case values from 2.0 to 5.0) for the conditions corresponding to the `Control` and `RecA` data. We can
do this by supplying different arguments to the plot function::

    fit.plot("Control", "k--", np.arange(2.0, 5.0, 0.01))
    fit.plot("RecA", "k--", np.arange(2.0, 5.0, 0.01))

Or what if we really only want the model prediction, then we can do::

    fit.plot("Control", "k--", np.arange(2.0, 5.0, 0.01), plot_data=False)

It is also possible to obtain simulations from the model directly. We can do this by calling the model with values for
the independent variable (here denoted as distance) and the parameters required to simulate the model. We obtain these
parameters by grabbing them from our fit object using the data handles::

    distance = np.arange(2.0, 5.0, 0.01)
    simulation_result = model(distance, fit["Control"])

Basically what happens here is that `fit["Control"]` grabs those parameters needed to simulate the condition
corresponding to the dataset with the name `control`. By providing specifically those parameters to the model, we can
simulate that condition.

Incremental fitting
-------------------

Fits can also be done incrementally::

    >>> model = lk.inverted_odijk("DNA")
    >>> fit = lk.FdFit(model)
    >>> print(fit.params)
    No parameters

We can see that there are no parameters to be fitted. The reason for this is that we did not add any data to the fit
yet. Let's add some and fit this data::

    >>> fit.add_data("Control", f1, d1)
    >>> fit.fit()
    >>> print(fit.params)
    Name         Value  Unit      Fitted      Lower bound    Upper bound
    ------  ----------  --------  --------  -------------  -------------
    DNA/Lp    59.409    [nm]      True                  0            inf
    DNA/Lc     2.81072  [micron]  True                  0            inf
    DNA/St  1322.9      [pN]      True                  0            inf
    kT         4.11     [pN*nm]   False                 0              8

Let's add a second data set where we expect a different contour length and refit::

    >>> fit.add_data("RecA", f2, d2, params={"DNA/Lc": "DNA/Lc_RecA"})
    >>> print(fit.params)
    Name              Value  Unit      Fitted      Lower bound    Upper bound
    -----------  ----------  --------  --------  -------------  -------------
    DNA/Lp         89.3347   [nm]      True                  0            inf
    DNA/Lc          2.80061  [micron]  True                  0            inf
    DNA/St       1597.68     [pN]      True                  0            inf
    kT              4.11     [pN*nm]   False                 0              8
    DNA/Lc_RecA     3.7758   [micron]  True                  0            inf
    
We see that indeed the second parameter now appears. We also note that the parameters from the first fit changed. If
this was not intentional, we should have fixed these parameters after the first fit. For example, we can fix the
parameter `DNA/Lp` by invoking::

    >>> fit["DNA/Lp"].fixed = True
    

Calculating per point contour length
------------------------------------

Sometimes, one wishes to invert the model with respect to one parameter (i.e. re-estimate one parameter on a per data
point basis). This can be used to obtain dynamic contour lengths for instance. In pylake, such an analysis can easily
be performed. We first set up a model and fit it to some data. This is all analogous to what we've learned before::

    # Define the model to be fitted
    model = lk.inverted_odijk("model") + lk.force_offset("model")

    # Fit the overall model first
    fit = lk.FdFit(model)
    fit.add_data("Control", force, distance)
    fit.fit()

Now, we wish to allow the contour length to vary on a per data point basis. For this, we use the function
`parameter_trace`. Here we see a few things happening. The first argument specifies the model to use for the inversion.

The second argument should contain the parameters to be used in this method. Note how we select them from the parameters
in the `fit` using the same syntax as before (i.e. `fit[data_name]`). Next, we specify which parameter has to be fitted
on a per data point basis. This is the parameter that we will re-estimate for every data point. Finally, we supply the
data to use in this analysis. First the independent parameter is passed, followed by the dependent parameter::

    lcs = lk.parameter_trace(model, fit["Control"], "model/Lc", distance, force)
    plt.plot(lcs)

The result of this analysis is an estimated contour length per data point, which can be used in subsequent analyses.

Advanced usage
--------------

Adding many data sets
*********************

Sometimes, you may want to add a large number of data sets with different offsets. Consider two lists of distance and
force vectors stored in `distances` and `forces`. In this case, it may make sense to load them in a loop and set such
transformations programmatically. We can iterate over both lists at once by using `zip`. In addition, we wanted to have
a different offset for each data set. This means that we'd need to give those new offsets a name. Let's just number
them. By adding enumerate, we also obtain an iteration counter, which we store in `i`. The whole procedure can then
succinctly be summarized in just two lines of code::

    for i, (d, f) in enumerate(zip(distances, forces)):
        fit.add_data(f"RecA {i}", f, d, params={"DNA/f_offset": f"DNA/f_offset_{i}"})

The syntax `f"DNA/f_offset_{i}"` is parsed into `DNA/f_offset_0`, `DNA/f_offset_1` ... etc. For more information on
how this works, read up on Python fantastic f-Strings.

Global fits versus single fits
******************************

The `FdFit` object manages a fit. To illustrate its use, and how a global fit differs from a local fit, consider the
following two examples::

    model = lk.inverted_odijk("DNA")
    fit = lk.FdFit(model)
    for i, (distance, force) in enumerate(zip(distances, forces)):
        fit.add_data(f"RecA {i}", f=force, d=distance)
    fit.fit()
    print(fit["DNA/Lc"])

and::

    for i, (distance, force) in enumerate(zip(distances, forces)):
        model = lk.inverted_odijk("DNA")
        fit = lk.FdFit(model)
        fit.add_data(f"RecA {i}", f=force, d=distance)
        fit.fit()
        print(fit["DNA/Lc"])

The first example is what we refer to as a global fit whereas the second example is an example of a local fit. The
difference between these two is that the former sets up one model that has to fit all the data whereas the latter fits
all the data sets independently. The former has one parameter set, whereas the latter has a parameter set per data set.
Also note how in the second example a new `Model` and `FdFit` is created at every cycle of the for loop.

Statistically, it is typically more optimal to fit data using global fitting (meaning you use one model to fit all the
data, as opposed to recreating the model for each new set of data), as more information goes into estimates of
parameters shared between different conditions. It's usually a good idea to think about which parameters you expect to
be different between different experiments and only allow these parameters to be different in the fit. For example,
if the only expected difference between the experiments is the contour length, then this can be achieved using::

    model = lk.inverted_odijk("DNA")
    fit = lk.FdFit(model)
    for i, (distance, force) in enumerate(zip(distances, forces)):
        fit.add_data(f"RecA {i}", force, distance, {"DNA/Lc": f"DNA/Lc_{i}"})
    fit.fit()
    print(fit.params)

Note that this piece of code will lead to parameters `DNA/Lc_0`, `DNA/Lc_1` etc.

Multiple models
***************

When working with multiple models, things can get a little more complicated. Let's say we have two models, `model1` and
`model2` and we want to fit both in a global fit. Constructing the `FdFit` is easy::

    model1 = lk.inverted_odijk("DNA")
    model2 = (lk.odijk("DNA") + lk.odijk("protein")).invert()
    fit = lk.FdFit(model1, model2)

But then the question arises, how do we add data to each model? Well, the trick is in the assignments to `model1` and
`model2`. We can use these now to add data to each model as follows::

    fit[model1].add_data("data for model 1", forces_1, distances_1)
    fit[model2].add_data("data for model 2", forces_2, distances_2)

See how we used the model handles? They are used to let the `FdFit` know to which model each data set should be added.
You can add as many data sets as you want to both models and fit it all at once.

Plotting is straightforward in this setting. We can plot the data sets corresponding to model 1 and 2 as follows::

    fit[model1].plot()
    fit[model2].plot()

Accessing the model parameters for a specific data set is a little more complicated in this setting. If we want to
obtain the parameters for "data for model 1", we'd have to invoke::

    params = fit[model1]["data for model 1"]

Note how we are now forced to index the model first using the square brackets, and only then access the data set by
name. An unfortunate necessity when it comes to multi-model curve fitting.


Confidence intervals and standard errors
****************************************

Once parameters have been fitted, standard errors can easily be obtained as follows::

    fit["DNA/Lc"].stderr

Assuming that the parameters are not at the bounds, the sum of random variables with finite moments converges to a
Gaussian distribution. This allows for the computation of confidence intervals using the Wald test
:cite:`press1990numerical`. To get these asymptotic intervals, we can use the member function `.ci` with a desired
confidence interval::

    fit["DNA/Lc"].ci(0.95)

Note that the bounds returned by this call are only asymptotically correct and should be used with caution. Better
confidence intervals can be obtained using the profile likelihood method :cite:`raue2009structural,maiwald2016driving`.
Note that these profiles require iterative computation and are therefore time consuming to produce. Determining
confidence intervals via profiles has two big advantages however:

- The confidence intervals no longer depend on the parametrization of the model (for more information on this see :cite:`maiwald2016driving`).
- By inspecting the profile, we can diagnose problems with the model we are using.

Profiles can easily be computed by calling :func:`~lumicks.pylake.FdFit.profile_likelihood` on the fit::

    profile = fit.profile_likelihood("DNA/Lc", num_steps=1000)

For a well parametrized model with sufficient data, a profile plot results in a (near) parabolic shape, where the line
of the parabola intersects with the confidence interval lines (dashed). The confidence intervals are then determined to
be at those intersection points::

    profile.plot()

.. image:: profile_good.png

One thing that may be of interest is to plot the relations between parameters in these profile likelihoods::

    profile.plot_relations()

These inferred relations can provide information on the coupling between different parameters. This can be quite
informative when diagnosing fitting issues. For example, when fitting a contour length in the presence of an distance
offset, we can observe that the two are related. To produce the following figure, we set a lower bound and upper bound
of -0.1 and 0.1 for the distance respectively. We can see that the profile is perfectly flat until the distance reaches
the bound. Only then does the profile suddenly jump.

.. image:: profile_bad.png

What this shows is that a change in one parameter (`DNA/Lc_RecA`) can be compensated by a change in the other. This
highlights the importance of constraining distance offset parameters when trying to estimate an absolute contour length.
In this sample case, fixing the distance offset to zero recovers the parabolic profile from before.
