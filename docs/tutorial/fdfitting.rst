FD Fitting
==========

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

Pylake introduces functionality to be able to fit force extension curves with various 
polymer models. This tutorial aims at being a gentle introduction to these fitting 
capabilities. Each section will first show the more basic mode of use, followed by more
advanced ways of interacting with the models, data and plots.

Models
------

When fitting data, everything revolves around models. One of these models is the so-called
extensible worm-like-chain model by Odijk et al. Let's have a look at it. We can construct
this model using the supplied pylake function odijk.

Note that we also have to give it a name. This name will be prefixed to model specific
parameters in this model. So for instance, the persistence length in this model, would be
named `DNA_Lp`. Let's have a look at the model and its parameters::

    >>> odijk_forward = pylake.odijk("DNA")

    Model equation:

    d(f) = DNA_Lc * (1 - (1/2)*sqrt(kT/(f*DNA_Lp)) + f/DNA_St)

    Parameter defaults:

    Name      Value  Unit      Fitted      Lower bound    Upper bound
    ------  -------  --------  --------  -------------  -------------
    DNA_Lp    40     [nm]      True                  0            inf
    DNA_Lc    16     [micron]  True                  0            inf
    DNA_St  1500     [pN]      True                  0            inf
    kT         4.11  [pN*nm]   False                 0              8

Note how most of the parameters listed have the model name prefixed. The only parameter
that doesn't is a parameter that is not expected to be model specific (namely the
parameter representing the Boltzmann constant times the temperature). We can also
obtain these parameters as a list::

    >>> print(odijk_forward.parameter_names)
        ['DNA_Lp', 'DNA_Lc', 'DNA_St', 'kT']

As we can see from the model equation, the model is given as distance as a function
of force.


Model composition and inversion
-------------------------------

In practice, we would typically want to fit force as a function of distance however. For this
we have the inverted Odijk model::


    odijk_inverted = pylake.inverted_odijk("DNA")


We can have a quick look at what this model looks like and which parameters are in there::

    >>> pylake.inverted_odijk("DNA")

    Model equation:

    f(d) = argmin[f](norm(DNA_Lc \left(1 - \frac12\sqrt{\frac{kT}{f DNA_Lp}} + \frac{f}{DNA_St}\right) - d))

    Parameter defaults:

    Name      Value  Unit      Fitted      Lower bound    Upper bound
    ------  -------  --------  --------  -------------  -------------
    DNA_Lp    40     [nm]      True                  0            inf
    DNA_Lc    16     [micron]  True                  0            inf
    DNA_St  1500     [pN]      True                  0            inf
    kT         4.11  [pN*nm]   False                 0              8

This seems to be what we want. Force as a function of distance. Now let's say we have acquired data,
but we notice that the template matching wasn't completely optimal. And that we had some force drift,
while forgetting to reset the force back to zero. In this case, we can incorporate offsets in our
model. We can introduce an offset in the independent parameter, by calling `subtract_independent_offset`
on our model::

    odijk_with_offset = pylake.inverted_odijk("DNA").subtract_independent_offset("d_offset")

If we also expect an offset in the dependent parameter, we can simply add an offset model to our
model::

    odijk_with_offsets = pylake.inverted_odijk("DNA").subtract_independent_offset("d_offset") + pylake.force_offset("f")

From the above example, you can see how easy it is to composite models. Sometimes, models become more 
complicated. For instance, we may have two worm like chain models that we wish to add, and then invert.
For the Odijk model, this can be done as follows::

    two_odijk = (pylake.odijk("DNA") + pylake.odijk("protein") + pylake.force_offset("f")).invert()


Note how we added three models and then inverted the composition of those models. The parentheses 
are important here, since otherwise we would have only inverted the offset model. Note that models
inverted via `invert()` will typically be slower than the pre-inverted counterparts. This is because
the inversion is done numerically, rather than analytically. For more complex examples on how this
inversion may be used, please see the examples.

For a full list of models that are available, please refer to the documentation by invoking
`help(pylake.fitting.models)`.


Loading data
------------

Next up, is loading some data. Let's assume we have two data sets. One was acquired in the presence
of a ligand, and another was measured without a ligand. We expect this ligand to only affect the 
contour length of our DNA. Loading the first dataset is simple. Note that we explicitly define
which dataset is force and which is distance via keyed arguments (this is intended to prevent
mistakes in argument order)::

    data1 = odijk_with_offsets.load_data(f=force1, d=distance1, name="Control")

Note how load_data returns a handle to the loaded data. We store this in `data1`. Such handles 
contain information about which parameters are used in that simulation condition. They can be
used to get more fine grained control over what is plotted or simulated. More on that later in
this tutorial. For the second dataset, we want the contour length to be different. We can achieve
this by renaming it when loading the data::

    data2 = odijk_with_offsets.load_data(f=force2, d=distance2, name="RecA", DNA_Lc="DNA_Lc_RecA")

More specifically, we renamed the parameter `DNA_Lc` to `DNA_Lc_RecA`. Sometimes, you may want
a large number of datasets with different offsets. Assuming we have two lists of distance and
force vectors stored in the lists distances and forces. In this case, it may make sense to load
them in a loop and set such transformations programmatically::

    for i, (distance, force) in enumerate(zip(distances, forces)):
        odijk_with_offsets.load_data(f=force, d=distance, name="RecA", f_offset=f"f_offset_{i}")

The syntax `f"offset_{i}"` is parsed into `offset_0`, `offset_1` ... etc. For more information
on how this works, read up on Python fantastic f-Strings.

Fitting the data
----------------

Once the data loaded, we can fit the data. To do this, we have to create a `FitObject`. This 
object will collect all the parameters involved in the models and data, and will allow you to 
interact with the model parameters and fit them. We construct it using `pylake.FitObject` and 
passing it one or more models. In return, we get an object we can interact with, which in this
case we store in `odijk_fit`::

    odijk_fit = pylake.FitObject(odijk_with_offsets)

The parameters of the model can be accessed under `parameters`. Note that by default, parameters 
tend to have reasonable initial guesses and bounds in pylake, but we can set our initial guess and
a lower and upper bound as follows::

    odijk_fit.parameters["DNA_Lp"].value = 50
    odijk_fit.parameters["DNA_Lp"].lb = 39
    odijk_fit.parameters["DNA_Lp"].ub = 80

After this, the model is ready to be fitted::

    odijk_fit.fit()

Note that multiple models can be fit at once, by just supplying more than one model::

    multi_model_fit = pylake.FitObject(model1, model2, model3)

Frequently, such a global fit has better statistical properties than fitting the data separately
as more information is available to infer parameters shared between the various models.


Plotting the data
-----------------

Fits can be plotted using the built-in plot functionality::
    
    odijk_fit.plot()
    plt.ylabel('Force [pN]')
    plt.xlabel('Distance [$\\mu$M]');

However, sometimes more fine grained control over the plots is required. Let's say we want to plot
the model over the range 2.0 to 5.0 for the conditions from `data1` and `data2`. We can do this by
calling plot on the model directly::

    dna_model.plot(odijk_fit.parameters[data1], np.arange(2.0, 5.0, .01), fmt='k--')
    dna_model.plot(odijk_fit.parameters[data2], np.arange(2.0, 5.0, .01), fmt='k--')

Note how we use the square brackets to select the parameters belonging to condition 1 and 2 using
the data handles `data1` and `data2` that we stored earlier. These collect the parameters relevant
for that particular experimental condition.

It is also possible to obtain simulations from the model directly. We can do this by calling the 
model with values for the independent variable (here denoted as distance) and the parameters 
required to simulate the model. Again, we obtain these parameters by grabbing them from our fit
object using the data handles::

    distance = np.arange(2.0, 5.0, .01)
    simulation_result = dna_model(distance, odijk_fit.parameters[data1])

Global fits versus single fits
------------------------------

The `FitObject` manages a fit. To illustrate its use, and how a global fit differs from a
simple fit, consider the following two examples::

    odijk_inv = pylake.inverted_odijk("DNA")
    for i, (distance, force) in enumerate(zip(distances, forces)):
        odijk_inv.load_data(f=force, d=distance, name="RecA")
    odijk_fit = pylake.FitObject(odijk_inv)
    odijk_fit.fit()
    print(odijk_fit.parameters["DNA_Lc"])

and::

    for i, (distance, force) in enumerate(zip(distances, forces)):
        odijk_inv = pylake.inverted_odijk("DNA")
        odijk_inv.load_data(f=force, d=distance, name="RecA")
        odijk_fit = pylake.FitObject(odijk_inv)
        odijk_fit.fit()
        print(odijk_fit.parameters["DNA_Lc"])

The difference between these two is that the former sets up a single model, that has to fit
all the data whereas the latter fits all the datasets independently. The former has one single
parameter set, whereas the latter has a parameter set per data set. Also note how in the second
example a new `Model` and `FitObject` is created at every cycle of the for loop.

Statistically, it is typically more optimal to fit data using global fitting, as more
information goes into estimates of parameters shared between different conditions. It's
usually a good idea to think about which parameters you expect to be different between
different experiments and only allow these parameters to be different. For example, if the
only expected difference between different experiments is the contour length, then this
can be achieved using::

    odijk_inv = pylake.inverted_odijk("DNA")
    for i, (distance, force) in enumerate(zip(distances, forces)):
        odijk_inv.load_data(f=force, d=distance, name="RecA", DNA_Lc=f"DNA_Lc_{i}")
    odijk_fit = pylake.FitObject(odijk_inv)
    odijk_fit.fit()
    print(odijk_fit.parameters)

Note that this piece of code will lead to parameters `DNA_Lc_0`, `DNA_Lc_1` etc.

Incremental fitting
-------------------

Fits can also be done incrementally::

    >>> odijk_inv = pylake.inverted_odijk("DNA")
    >>> odijk_fit = pylake.FitObject(odijk_inv)
    >>> print(odijk_fit.parameters)
    No parameters

We can see that there are no parameters to be fitted. The reason for this is that
we did not add any data to the model yet. Let's add some and fit this data::

    >>> data1 = odijk_inv.load_data(f=f1, d=d1, name="Control")
    >>> odijk_fit.fit()
    >>> print(odijk_fit.parameters)
    Name         Value  Unit      Fitted      Lower bound    Upper bound
    ------  ----------  --------  --------  -------------  -------------
    DNA_Lp    59.409    [nm]      True                  0            inf
    DNA_Lc     2.81072  [micron]  True                  0            inf
    DNA_St  1322.9      [pN]      True                  0            inf
    kT         4.11     [pN*nm]   False                 0              8

Let's add a second dataset where we expect a different contour length and refit::

    >>> data2 = odijk_inv.load_data(f=f2, d=d2, name="RecA", DNA_Lc="DNA_Lc_RecA")
    >>> print(odijk_fit.parameters)
    Name              Value  Unit      Fitted      Lower bound    Upper bound
    -----------  ----------  --------  --------  -------------  -------------
    DNA_Lp         89.3347   [nm]      True                  0            inf
    DNA_Lc          2.80061  [micron]  True                  0            inf
    DNA_St       1597.68     [pN]      True                  0            inf
    kT              4.11     [pN*nm]   False                 0              8
    DNA_Lc_RecA     3.7758   [micron]  True                  0            inf
    
We see that indeed the second parameter now appears. We also note that the parameters
from the first fit changed. If this was not intentional, we should have fixed
these parameters after the first fit. For example, we can fix the parameter `DNA_Lp`
by invoking::

    >>> odijk.fit.parameters["DNA_Lp"].vary = false
    

Calculating per point contour length
------------------------------------

Sometimes, one wishes to invert the model with respect to one parameter (i.e. re-estimate one 
parameter on a per data point basis). This can be used to obtain dynamic contour lengths for
instance. In pylake, such an analysis can easily be performed. We first set up a model and
fit it to some data. This is all analogous to what we've learned before::

    # Define the model to be fitted
    model = pylake.inverted_odijk("model") + pylake.force_offset("f", "offset")

    # Fit the overall model first
    data_handle = model.load_data(f=force, d=distance)
    current_fit = pylake.FitObject(model)
    current_fit.fit()

Now, we wish to allow the contour length to vary on a per data point basis. For this, we use
the function `parameter_trace`. Here we see a few things happening. The first argument is a model
to use for the inversion.

The second argument contains the parameters to use in this model. Note how we select them from
the parameters in the fit object using the same syntax as before (i.e.
`fitobject.parameters[data_handle]`). Next, we specify which parameter has to be fitted on a per
data point basis. This is the parameter that we will re-estimate for every data point. Finally,
we supply the data to use in this analysis. First the independent parameter is passed, followed
by the dependent parameter::

    lcs = parameter_trace(model, current_fit.parameters[data_handle], "model_Lc", distance, force)
    plt.plot(lcs)

The result is an estimated contour length per data point, which can be used in subsequent
analyses.
