DNA FD Fitting
==============

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

First we import our required libraries::

    import matplotlib.pyplot as plt
    import lumicks.pylake as pylake
    import numpy as np

Load the data from text files
-----------------------------

The next step is loading the data. In this case, the data is provided as
text files::

    d1 = np.genfromtxt('RecA/Bare pKYBI.txt', delimiter='\t')
    d1 = d1[2:-1, :]  # Drop the header
    f1 = d1[:, 2]
    d1 = d1[:, 1]
    
    d2 = np.genfromtxt('RecA/RecA pKYBI.txt', delimiter='\t')
    d2 = d2[2:-1, :]  # Drop the header
    f2 = d2[:, 2]
    d2 = d2[:, 1]

Set up the model
----------------

For this we want to use an inverted WLC model with an offset in force
and distance.

While it possible (and equivalent) to generate this model via
``pylake.force_model('DNA', 'WLC').invert()``, an optimized inverted WLC
model is explicitly included as a separate model. When only a single WLC
is needed, it is best to use this one, as it is considerably faster than
explicitly inverting::

    M_DNA = pylake.inverted_odijk("DNA").subtract_independent_offset("d_offset") + pylake.force_offset("f")

Let's have a look at the parameters in this model::

    >>> print(M_DNA.parameter_names)
    ['d_offset', 'DNA_Lp', 'DNA_Lc', 'DNA_St', 'kT', 'f_offset']

Load the data
-------------

Next, we load the data into the model. We're going to filter the data by
force, to make sure that we discard data outside of the model’s valid range.
We can do this by creating a logical mask which is true for the data we wish
to include and false for the data we wish to exclude::

    # Minimum and maximum force value to include
    f_min = 0
    f_max = 25
    
    f1_mask = np.logical_and(f1 < f_max, f1 > f_min)
    f2_mask = np.logical_and(f2 < f_max, f2 > f_min)

We can load data into the model by using the function `load_data`. Note
that we apply the mask as we are loading the data using the square brackets::

    data1 = M_DNA.load_data(f=f1[f1_mask], d=d1[f1_mask], name="Control")

If parameters are expected to differ between conditions, we can rename them
when loading the data. For the second data set, we expect the contour length
and persistence length to be different. Let's rename these::

    data2 = M_DNA.load_data(f=f2[f2_mask], d=d2[f2_mask], name="RecA", DNA_Lc="DNA_Lc_RecA", DNA_Lp="DNA_Lp_RecA")

Set up the fit
--------------

Now, let's fit our model to the data we just loaded. To do this, we have to create a FitObject::

    F = pylake.FitObject(M_DNA)
    
We would also like to set some parameter bounds::

    def set_bounds(F):
        F.parameters["DNA_Lp"].value = 50
        F.parameters["DNA_Lp"].lb = 39
        F.parameters["DNA_Lp"].ub = 80
    
        F.parameters["DNA_Lp_RecA"].value = 50
        F.parameters["DNA_Lp_RecA"].lb = 39
        F.parameters["DNA_Lp_RecA"].ub = 280
    
        F.parameters["DNA_St"].value = 1200
        F.parameters["DNA_St"].lb = 700
        F.parameters["DNA_St"].ub = 2000
        F.parameters["d_offset"].ub = 5
    
    set_bounds(F)
    F.fit();

Plot the fit and print the parameter values
-------------------------------------------

Plotting the fit alongside the data is easy. Simply call the plot function
on the fitobject (i.e. `F.plot()`). Similarly, we can print the parameters 
to a table by invoking `F.parameters`::

    F.plot()
    plt.ylabel('Force [pN]')
    plt.xlabel('Distance [$\\mu$M]');
    F.parameters

.. raw:: html

    <table>
    <thead>
    <tr><th>Name       </th><th style="text-align: right;">      Value</th><th>Unit    </th><th>Fitted  </th><th style="text-align: right;">  Lower bound</th><th style="text-align: right;">  Upper bound</th></tr>
    </thead>
    <tbody>
    <tr><td>d_offset   </td><td style="text-align: right;">  -0.299   </td><td>NA      </td><td>True    </td><td style="text-align: right;">         -inf</td><td style="text-align: right;">            5</td></tr>
    <tr><td>DNA_Lp     </td><td style="text-align: right;">  66.8239  </td><td>[nm]    </td><td>True    </td><td style="text-align: right;">           39</td><td style="text-align: right;">           80</td></tr>
    <tr><td>DNA_Lc     </td><td style="text-align: right;">   3.12586 </td><td>[micron]</td><td>True    </td><td style="text-align: right;">            0</td><td style="text-align: right;">          inf</td></tr>
    <tr><td>DNA_St     </td><td style="text-align: right;">2000       </td><td>[pN]    </td><td>True    </td><td style="text-align: right;">          700</td><td style="text-align: right;">         2000</td></tr>
    <tr><td>kT         </td><td style="text-align: right;">   4.11    </td><td>[pN*nm] </td><td>False   </td><td style="text-align: right;">            0</td><td style="text-align: right;">            8</td></tr>
    <tr><td>f_offset   </td><td style="text-align: right;">   0.287546</td><td>NA      </td><td>True    </td><td style="text-align: right;">            0</td><td style="text-align: right;">          inf</td></tr>
    <tr><td>DNA_Lp_RecA</td><td style="text-align: right;"> 238.134   </td><td>[nm]    </td><td>True    </td><td style="text-align: right;">           39</td><td style="text-align: right;">          280</td></tr>
    <tr><td>DNA_Lc_RecA</td><td style="text-align: right;">   4.04548 </td><td>[micron]</td><td>True    </td><td style="text-align: right;">            0</td><td style="text-align: right;">          inf</td></tr>
    </tbody>
    </table>


.. image:: output_10_2.png

Let's print some of our parameters. The parameter we are most interested in is
the contour length difference. We multiply by 1000 since we desire this value
in nanometers::

    M_DNA.plot(F.parameters, data1, 'r--', np.arange(2.1, 5.0, .01))
    M_DNA.plot(F.parameters, data2, 'r--', np.arange(2.1, 5.0, .01))
    plt.ylabel('Force [pN]')
    plt.xlabel('Distance [$\\mu$M]')
    plt.ylim([0, 30])
    plt.xlim([2, 4])
    
    print(f"Boltzmann * Temperature: {F.parameters['kT'].value:.2f} [pN nm]")
    print(f"Force offset: {F.parameters['f_offset'].value:.2f} [pN]")
    print(f"Distance offset: {F.parameters['d_offset'].value * 1000:.2f} [nm]")
    print(f"DNA persistence Length: {F.parameters['DNA_Lp'].value:.2f} [nm]")
    print(f"DNA contour Length: {F.parameters['DNA_Lc'].value * 1000:.2f} [nm]")
    print(f"Stretch modulus: {F.parameters['DNA_St'].value:.2f} [pN]")
    print(f"Contour length difference: {(F.parameters['DNA_Lc_RecA'].value - F.parameters['DNA_Lc'].value) * 1000:.2f} [nm]")
    print(f"Residual: {sum(F._calculate_residual()**2)}")


.. parsed-literal::

    Boltzmann * Temperature: 4.11 [pN nm]
    Force offset: 0.29 [pN]
    Distance offset: -299.00 [nm]
    DNA persistence Length: 66.82 [nm]
    DNA contour Length: 3125.86 [nm]
    Stretch modulus: 2000.00 [pN]
    Contour length difference: 919.62 [nm]
    Residual: 1151.327895904549
    

.. image:: output_11_2.png


Take a closer look at the fit
-----------------------------

To assess the model fidelity to the data, we can have a closer look at
the force extension curves::

    F.plot()
    plt.ylabel('Force [pN]')
    plt.xlabel('Distance [$\\mu$M]')
    plt.ylim([0, 5]);


.. image:: output_13_1.png


Include a data specific force offset
------------------------------------

We can see that there is some deviation between the model and the data.
It’s possible that there was a tiny force drift between the two
experiments. Let’s try including an extra parameter for the force offset
of the second condition. Let’s also try a few different models to fit
this data::

    M_DNA = pylake.inverted_odijk("DNA").subtract_independent_offset("d_offset") + pylake.force_offset("f")
    M_DNA_MS = pylake.marko_siggia_ewlc_force("DNA").subtract_independent_offset("d_offset") + pylake.force_offset("f")
    
    M_DNA.load_data(f=f1[f1_mask], d=d1[f1_mask], name="Control")
    M_DNA.load_data(f=f2[f2_mask], d=d2[f2_mask], name="RecA", DNA_Lc="DNA_Lc_RecA", DNA_Lp="DNA_Lp_RecA", f_offset="f_offset2")
    odijk_offset = pylake.FitObject(M_DNA)
    set_bounds(odijk_offset)
    odijk_offset.fit()
    
    M_DNA_MS.load_data(f=f1[f1_mask], d=d1[f1_mask], name="Control")
    M_DNA_MS.load_data(f=f2[f2_mask], d=d2[f2_mask], name="RecA", DNA_Lc="DNA_Lc_RecA", DNA_Lp="DNA_Lp_RecA", f_offset="f_offset2")
    siggia_offset = pylake.FitObject(M_DNA_MS)
    set_bounds(siggia_offset)
    siggia_offset.fit();

Plot the competing models
-------------------------

Now we can see that the fit is quite a bit better. We can also see that
the predictions for the contour length difference are quite similar,
increasing our confidence in our results::

    plt.figure(figsize=(20,5))
    plt.subplot(1, 2, 1)
    odijk_offset.plot()
    plt.title('Odijk')
    plt.ylim([0,10])
    plt.subplot(1, 2, 2)
    siggia_offset.plot()
    plt.title('Marko-Siggia')
    plt.ylim([0,10])
    
    print(f"Contour length difference Odijk: {(odijk_offset.parameters['DNA_Lc_RecA'].value - odijk_offset.parameters['DNA_Lc'].value) * 1000:.2f} [nm]")
    print(f"Contour length difference Marko-Siggia: {(siggia_offset.parameters['DNA_Lc_RecA'].value - siggia_offset.parameters['DNA_Lc'].value) * 1000:.2f} [nm]")


.. parsed-literal::

    Contour length difference Odijk: 911.70 [nm]
    Contour length difference Marko-Siggia: 913.09 [nm]
    


.. image:: output_17_1.png


Which fit is statistically optimal
----------------------------------

We can also determine how well a model fits the data by looking at the
corrected Akaike Information Criterion and Bayesian Information
Criterion. Here, a low value indicates a better model.

We can see here that both criteria seem to indicate that the
Marko-Siggia model with two offsets provides the best fit. Please note
however, that it is always important to verify that the model produce
sensible results. More freedom to fit parameters, will almost always
lead to an improved fit, and this additional freedom can lead to fits
that produce non-physical results.

Generally it is always a good idea to try multiple models, and multiple
sets of bound constraints, to get a feel for how reliable the estimates
are::

    print("Corrected Akaike Information Criterion")
    print(f"Odijk Model with single force offset {F.aicc}")
    print(f"Odijk Model with two force offsets {odijk_offset.aicc}")
    print(f"Marko-Siggia Model with two force offsets {siggia_offset.aicc}")
    
    print("Bayesian Information Criterion")
    print(f"Odijk Model with single force offset {F.bic}")
    print(f"Odijk Model with two force offsets {odijk_offset.bic}")
    print(f"Marko-Siggia Model with two force offsets {siggia_offset.bic}")


.. parsed-literal::

    Corrected Akaike Information Criterion
    Odijk Model with single force offset 7067.68118147101
    Odijk Model with two force offsets 6208.44389146499
    Marko-Siggia Model with two force offsets 6281.818847723742
    Bayesian Information Criterion
    Odijk Model with single force offset 7114.174380931719
    Odijk Model with two force offsets 6261.576151211682
    Marko-Siggia Model with two force offsets 6334.951107470434
    

Again, we also look at the parameters::

    F.parameters




.. raw:: html

    <table>
    <thead>
    <tr><th>Name       </th><th style="text-align: right;">      Value</th><th>Unit    </th><th>Fitted  </th><th style="text-align: right;">  Lower bound</th><th style="text-align: right;">  Upper bound</th></tr>
    </thead>
    <tbody>
    <tr><td>d_offset   </td><td style="text-align: right;">  -0.299   </td><td>NA      </td><td>True    </td><td style="text-align: right;">         -inf</td><td style="text-align: right;">            5</td></tr>
    <tr><td>DNA_Lp     </td><td style="text-align: right;">  66.8239  </td><td>[nm]    </td><td>True    </td><td style="text-align: right;">           39</td><td style="text-align: right;">           80</td></tr>
    <tr><td>DNA_Lc     </td><td style="text-align: right;">   3.12586 </td><td>[micron]</td><td>True    </td><td style="text-align: right;">            0</td><td style="text-align: right;">          inf</td></tr>
    <tr><td>DNA_St     </td><td style="text-align: right;">2000       </td><td>[pN]    </td><td>True    </td><td style="text-align: right;">          700</td><td style="text-align: right;">         2000</td></tr>
    <tr><td>kT         </td><td style="text-align: right;">   4.11    </td><td>[pN*nm] </td><td>False   </td><td style="text-align: right;">            0</td><td style="text-align: right;">            8</td></tr>
    <tr><td>f_offset   </td><td style="text-align: right;">   0.287546</td><td>NA      </td><td>True    </td><td style="text-align: right;">            0</td><td style="text-align: right;">          inf</td></tr>
    <tr><td>DNA_Lp_RecA</td><td style="text-align: right;"> 238.134   </td><td>[nm]    </td><td>True    </td><td style="text-align: right;">           39</td><td style="text-align: right;">          280</td></tr>
    <tr><td>DNA_Lc_RecA</td><td style="text-align: right;">   4.04548 </td><td>[micron]</td><td>True    </td><td style="text-align: right;">            0</td><td style="text-align: right;">          inf</td></tr>
    </tbody>
    </table>



Let's see if the parameters for the other model are similar::

    siggia_offset.parameters




.. raw:: html

    <table>
    <thead>
    <tr><th>Name       </th><th style="text-align: right;">        Value</th><th>Unit    </th><th>Fitted  </th><th style="text-align: right;">  Lower bound</th><th style="text-align: right;">  Upper bound</th></tr>
    </thead>
    <tbody>
    <tr><td>d_offset   </td><td style="text-align: right;">  -0.16129   </td><td>NA      </td><td>True    </td><td style="text-align: right;">         -inf</td><td style="text-align: right;">            5</td></tr>
    <tr><td>DNA_Lp     </td><td style="text-align: right;">  53.5762    </td><td>[nm]    </td><td>True    </td><td style="text-align: right;">           39</td><td style="text-align: right;">           80</td></tr>
    <tr><td>DNA_Lc     </td><td style="text-align: right;">   2.99474   </td><td>[micron]</td><td>True    </td><td style="text-align: right;">            0</td><td style="text-align: right;">          inf</td></tr>
    <tr><td>DNA_St     </td><td style="text-align: right;">2000         </td><td>[pN]    </td><td>True    </td><td style="text-align: right;">          700</td><td style="text-align: right;">         2000</td></tr>
    <tr><td>kT         </td><td style="text-align: right;">   4.11      </td><td>[pN*nm] </td><td>False   </td><td style="text-align: right;">            0</td><td style="text-align: right;">            8</td></tr>
    <tr><td>f_offset   </td><td style="text-align: right;">   5.3927e-19</td><td>NA      </td><td>True    </td><td style="text-align: right;">            0</td><td style="text-align: right;">          inf</td></tr>
    <tr><td>DNA_Lp_RecA</td><td style="text-align: right;"> 233.319     </td><td>[nm]    </td><td>True    </td><td style="text-align: right;">           39</td><td style="text-align: right;">          280</td></tr>
    <tr><td>DNA_Lc_RecA</td><td style="text-align: right;">   3.90783   </td><td>[micron]</td><td>True    </td><td style="text-align: right;">            0</td><td style="text-align: right;">          inf</td></tr>
    <tr><td>f_offset2  </td><td style="text-align: right;">   0.397233  </td><td>NA      </td><td>True    </td><td style="text-align: right;">            0</td><td style="text-align: right;">          inf</td></tr>
    </tbody>
    </table>



