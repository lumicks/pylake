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
extensible worm-like-chain model by Odijk et al. The Odijk model is given by the function:

    .. math:: d = Lc \left(1 - \frac{1}{2} \sqrt{\frac{k_B T}{F L_p}} + \frac{F}{S_t} \right) 

As we can see, it is given as distance as a function of force. We would like to fit the inverse 
model however. We can construct this model using the supplied pylake function odijk. Note that 
we also have to give it a name::


    odijk_forward = pylake.odijk("DNA")


This name will be prefixed to parameters in this model. So for instance, the persistence length
in this model, would be named `DNA_Lp`.


In practice, we would typically want to fit force as a function of distance however. For this 
we have the inverted Odijk model::


    odijk_inverted = pylake.inverted_odijk("DNA")


Now let's say we have acquired data, but we notice that the template matching wasn't completely 
optimal. And that we had some force drift, while forgetting to reset the force back to zero.


In this case, we would like to incorporate two offsets in our model. We can introduce an offset 
in the dependent parameter, by calling `subtract_offset` on our model::


    odijk_with_offset = pylake.inverted_odijk("DNA").subtract_offset("d_offset")


Now all that remains is to add the dependent offset to the model::


    odijk_with_offsets = pylake.inverted_odijk("DNA").subtract_offset("d_offset") + pylake.offset("f")


From the above example, you can see how easy it is to composite models. Sometimes, models become more 
complicated. For instance, we may have two worm like chain models that we wish to add, and then invert.
For the Odijk model, this can be done as follows::


    two_odijk = (pylake.odijk("DNA") + pylake.odijk("protein") + pylake.offset("f")).invert()


Note how we added three models, and then inverted the composition of those models. The parentheses 
are important here, since otherwise we would have only inverted the offset model. For more complex 
examples, please see the examples.


Loading data
------------

Next up, is loading some data. Let's assume we have two datasets. One was acquired in the presence 
of a ligand, and another was measured without a ligand. We expect this ligand to only affect the 
contour length of our DNA. We can load this data as follows::

    data1 = odijk_with_offsets.load_data(distance1, force1, name="Control")
    data2 = odijk_with_offsets.load_data(distance2, force2, name="RecA", DNA_Lc="DNA_Lc_RecA")

Note how load_data returns a handle to the loaded data. We store these in `data1` and `data2`.
These handles store which parameters are used in that simulation condition and can be used later 
to get more fine grained control over what is plotted or simulated.

Note that for the second dataset we renamed the parameter `DNA_Lc` to `DNA_Lc_RecA`. This signifies
that this parameter has to be different for this dataset. Sometimes, you may want a large number 
of datasets with different offsets. Assuming we have two lists of distance and force vectors stored
in the lists distances and forces. In this case, it may make sense to load them in a loop::

    for i, (distance, force) in enumerate(zip(distances, forces)):
        odijk_with_offsets.load_data(distance, force, name="RecA", DNA_Lc=f"offset_{i}")

The syntax `f"offset_{i}"` is parsed into `offset_0`, `offset_1` ... etc.

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

After this, the model can be fitted::

    odijk_fit.fit()

Note that multiple models can be fit at once, by just supplying more than one model::

    multi_model_fit = pylake.FitObject(model1, model2, model3)

Frequently, such a global fit has better statistical properties than fitting the data separately
as more information is available to infer parameters shared by the various models.

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

Note how we pass the handles `data1` and `data2` that we stored earlier to let pylake know which
conditions we want to plot the model for. They are used to collect the parameters relevant for
that particular experimental condition.

It is also possible to obtain simulations from the model directly. We can do this by calling the 
model with values for the independent variable (here denoted as distance) and the parameters 
required to simulate the model. We can obtain these parameters by grabbing them from our fit object
using the data handles::

    distance = np.arange(2.0, 5.0, .01)
    simulation_result = dna_model(distance, odijk_fit.parameters[data1])
