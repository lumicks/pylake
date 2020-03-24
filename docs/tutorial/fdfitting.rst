FD Fitting
==========

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

Pylake introduces functionality to be able to fit force extension curves with various 
polymer models. This tutorial aims at being a gentle introduction to these fitting 
capabilities.


When fitting data, everything revolves around models. One of these models is the so-called
extensible worm-like-chain model by Odijk et al. The Odijk model is given by the function:

    .. math:: d = Lc \left(1 - \frac{1}{2} \sqrt{\frac{k_B T}{F L_p}} + \frac{F}{S_t} \right) 

As we can see, it is given as distance as a function of force. We would like to fit the inverse 
model however. We can construct this model using the supplied pylake function odijk. Note that 
we also have to give it a name.

.. code:: ipython3

    odijk_forward = pylake.odijk("DNA")

This name will be prefixed to parameters in this model. So for instance, the persistence length
in this model, would be named `DNA_Lp`.


In practice, we would typically want to fit force as a function of distance however. For this 
we have the inverted Odijk model.

.. code:: ipython3

    odijk_inverted = pylake.inverted_odijk("DNA")

Now let's say we have acquired data, but we notice that the template matching wasn't completely 
optimal. And that we had some force drift, while forgetting to reset the force back to zero.


In this case, we would like to incorporate two offsets in our model. We can introduce an offset 
in the dependent parameter, by calling `subtract_offset` on our model:

.. code:: ipython3

    odijk_with_offset = pylake.inverted_odijk("DNA").subtract_offset("d_offset")

Now all that remains is to add the dependent offset to the model.

.. code:: ipython3

    odijk_with_offsets = pylake.inverted_odijk("DNA").subtract_offset("d_offset") + pylake.offset("f")

From the above example, you can see how easy it is to composite models. Sometimes, models become more 
complicated. For instance, we may have two worm like chain models that we wish to add, and then invert.
For the Odijk model, this can be done as follows:

.. code:: ipython3

    two_odijk = (pylake.odijk("DNA") + pylake.odijk("protein") + pylake.offset("f")).invert()

Note how we added three models, and then inverted the whole at once. The brackets are important here,
since otherwise we would have only inverted the offset model. For more complex examples, please 
see the examples.


Next up, is loading some data. Let's assume we have two datasets. One was acquired in the presence 
of a ligand, and another was measured without a ligand. We expect this ligand to only affect the 
contour length of our DNA. We can load this data easily as follows:

.. code:: ipython3

    odijk_with_offsets.load_data(distance1, force1, name="Control")
    odijk_with_offsets.load_data(distance2, force2, name="RecA", DNA_Lc="DNA_Lc_RecA")

Note that for the second dataset we renamed the parameter `DNA_Lc` to `DNA_Lc_RecA`. This signifies
that this parameter has to be different for this dataset.

With the data loaded, we can fit the data. To do this, we have to create a `FitObject`. This 
object will collect all the parameters involved in the models and data, and will allow you to 
interact with the model parameters and fit them. We construct it using `pylake.FitObject` and 
passing it one or more models. In return, we get an object we can interact with, which in this
case we store in `odijk_fit`. 

.. code:: ipython3

    odijk_fit = pylake.FitObject(odijk_with_offsets)

The parameters of the model can be accessed under `parameters`. Note that by default, parameters 
tend to have reasonable initial guesses and bounds in pylake, but we can set our initial guess and 
a lower and upper bound as follows:

.. code:: ipython3

    odijk_fit.parameters["DNA_Lp"].value = 50
    odijk_fit.parameters["DNA_Lp"].lb = 39
    odijk_fit.parameters["DNA_Lp"].ub = 80

After this, the model can be fitted and plotted.

.. code:: ipython3

    odijk_fit.fit()
    odijk_fit.plot()
    plt.ylabel('Force [pN]')
    plt.xlabel('Distance [$\\mu$M]');
