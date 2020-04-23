Twistable Worm-Like-Chain Fitting
=================================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

First we import the necessary libraries::

    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    from lumicks import pylake

Load the data from disk
-----------------------

This is pre Bluelake data, therefore we have to extract the data from the h5
file ourselves. Considering the valid range of the force model, we only consider
data below 60 piconewton::

    file = h5py.File("tWLC_data/20140206-155802 050Mg 500NaCl #018-001.h5", 'r')
    data = file['FdtData'][()]
    f = data[0, :]
    d = data[1, :]
    d = d[f < 60]
    f = f[f < 60]

Set up the model
----------------

Set up a twistable worm like chain model with a distance and force offset. By
default, the `twistable_wlc` model provided with pylake is defined as distance
as a function of force. Typically, we want to fit force as a function of distance
however. To achieve this, we can invert the model. In addition, we incorporate
an offset in both distance and force to compensate for small offsets that may
exist in the data::

    M_DNA = (pylake.twistable_wlc('DNA') + pylake.distance_offset('d')).invert() + pylake.force_offset('f')

Load the data into the model
----------------------------

We load the data into the model. This is done using the command `load_data`. Note
that we also supply a name for the data, which will be used when plotting for
example.

After this, we would like to fit the model to this data. To do this, we create
a `pylake.Fit`. These are used to keep track of the fitted parameters and optionally
fit multiple models at once. In this example, we only fit a single model::

    M_DNA.load_data(f=f, d=d, name="Twistable WLC")
    F = pylake.Fit(M_DNA)

Fit the model
-------------

Now we are ready to fit the model. Considering that the tWLC model is
expensive to evaluate, this may take a while. This is also why we choose
to enable verbose output::

    >>> F.fit(verbose=2)
    >>> plt.show()

       Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality   
           0              1         5.3647e+04                                    2.83e+08    
           1              2         2.3267e+02      5.34e+04       9.42e+00       2.19e+06    
           2              3         2.1138e+01      2.12e+02       1.79e+01       6.94e+04    
           3              4         2.0693e+01      4.45e-01       1.20e+02       3.32e+05    
           4              5         2.0125e+01      5.68e-01       3.17e+01       6.57e+04    
           5              6         2.0078e+01      4.70e-02       3.07e+01       9.32e+03    
           6             10         1.9967e+01      1.10e-01       7.96e+00       7.33e+03    
           7             15         1.9958e+01      8.95e-03       1.26e-01       2.62e+03    
           8             17         1.9955e+01      3.05e-03       6.35e-02       2.12e+03    
           9             20         1.9955e+01      2.39e-04       7.94e-03       1.23e+03    
          10             23         1.9955e+01      5.18e-06       4.97e-04       1.32e+03    
          11             25         1.9955e+01      6.56e-06       3.19e-05       1.19e+03    
          12             27         1.9955e+01      7.59e-07       3.06e-05       1.19e+03    
          13             28         1.9955e+01      4.58e-08       6.13e-05       1.33e+03    
          14             29         1.9955e+01      1.26e-06       3.87e-06       1.18e+03    
    `xtol` termination condition is satisfied.
    Function evaluations 29, initial cost 5.3647e+04, final cost 1.9955e+01, first-order optimality 1.18e+03.
    
Plotting the results
--------------------

After fitting we can plot our results and print our parameters. Doing this
is as simple as invoking `F.plot()` and `F.parameters`::

    plt.figure(figsize=(10,10))
    F.plot()
    plt.xlabel('Distance [$\\mu$m]')
    plt.ylabel('Force [pN]');


.. image:: output_9_1.png

We can also show the parameters::

    >>> F.parameters

    Name              Value  Unit        Fitted      Lower bound    Upper bound
    --------  -------------  ----------  --------  -------------  -------------
    DNA_Lp      42.7093      [nm]        True                  0            inf
    DNA_Lc      15.4259      [micron]    True                  0            inf
    DNA_St    1460.49        [pN]        True                  0            inf
    DNA_C      346.338       [pN*nm**2]  True                  0          50000
    DNA_g0    -638.638       [pN*nm]     True             -50000          50000
    DNA_g1      16.3832      [nm]        True             -50000          50000
    DNA_Fc      34.3838      [pN]        True                  0          50000
    kT           4.11        [pN*nm]     False                 0              8
    d_offset     1.077       NA          True                  0            inf
    f_offset     0.00503963  NA          True                  0            inf
