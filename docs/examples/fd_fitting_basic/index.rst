.. warning::
    This is beta functionality. While usable, this is a beta-functionality which
    has not yet been tested in a sufficient number of different scenarios. The API
    may also still be subject to change.

Twistable Worm-Like-Chain Fitting
=================================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

First we import the necessary libraries::

    import numpy as np
    import matplotlib.pyplot as plt
    from lumicks import pylake

Load the data from disk
-----------------------

Let's load and plot the data first::

    file = pylake.File('twlc_data//20200430-163932 FD Curve FD_1_control_forw.h5')
    fd_curve = file.fdcurves["FD_1_control_forw"]
    fd_curve.plot_scatter()

.. image:: output_9_1.png

Set up a basic model first
--------------------------

We clearly see that the force starts levelling out at high forces in the data. This
is a clear sign that to be able to describe this data, we'll need something more complex
in order to capture this behaviour we see in the data. The twistable worm like chain
is one such model that can describe the untwisting of DNA at high forces. However, its
complexity also incurs some challenges.

Parameter optimization always begins from an initial guess, and if this initial guess
is bad, it can get stuck at an estimated set of parameters that are suboptimal, a
so-called local optimum. One way to mitigate this, is to start at good initial values.

In this notebook, we fit the low part of the curve first, with a regular worm like
chain model, and then use those estimates as initial guesses to fit the twistable model.

Let's set up the Odijk model and create the fit::

    m_odijk = pylake.inverted_odijk('DNA').subtract_independent_offset() + pylake.force_offset('DNA')
    fit_odijk = pylake.FdFit(m_odijk)

Considering that this model only describes the lower part of the curve, we have to
extract the data that is relevant to us. We can obtain this data from the ``FdCurve``
as follows::

    force = fd_curve.f.data
    distance = fd_curve.d.data

We only wish to use the forces below 40, so we filter the data according to this
requirement::

    mask = force < 40
    distance = distance[mask]
    force = force[mask]

Now we are ready to add this data to the fit::

    fit_odijk.add_data("Inverted Odijk", force, distance)
    fit_odijk["DNA/d_offset"].upper_bound = .1
    fit_odijk["DNA/d_offset"].lower_bound = -.1

And fit the model::

    >>> fit_odijk.fit()

    Fit
      - Model: DNA(x-d)_with_DNA
      - Equation:
          f(d) = argmin[f](norm(DNA.Lc * (1 - (1/2)*sqrt(kT/(f*DNA.Lp)) + f/DNA.St)-(d - DNA.d_offset))) + DNA.f_offset

      - Data sets:
        - FitData(Inverted Odijk, N=959)

      - Fitted parameters:
        Name                 Value  Unit      Fitted      Lower bound    Upper bound
        ------------  ------------  --------  --------  -------------  -------------
        DNA/d_offset     0.102911   NA        True             -inf            inf
        DNA/Lp          43.4116     [nm]      True                0            100
        DNA/Lc           2.68676    [micron]  True                0            inf
        DNA/St        1554.16       [pN]      True                0            inf
        kT               4.11       [pN*nm]   False               0              8
        DNA/f_offset     0.0624994  [pN]      True               -0.1            0.1

Set up the Twistable worm like chain model
------------------------------------------

Set up a twistable worm like chain model with a distance and force offset. By default,
the `twistable_wlc` model provided with pylake is defined as distance as a function of
force. Typically, we want to fit force as a function of distance however. To achieve
this, we can invert the model using its `invert` function. However, this can be slow
in certain cases, as it requires an inversion for each data point. Luckily we have a
faster way of achieving this in pylake, which is to use the dedicated `inverted_twistable_wlc`
model.

Again, we incorporate an offset in both distance and force to compensate for small
offsets that may exist in the data:::

    m_dna = pylake.inverted_twistable_wlc('DNA').subtract_independent_offset() + pylake.force_offset('DNA')
    fit_twlc = pylake.FdFit(m_dna)

Load the data into the model
----------------------------

Now it is time to load the full data into the model. Note in the figure however,
that there seems to be a small break at the end of the Fd curve. The model will
not be able to capture this behaviour, and therefore it is best to remove it
prior to fitting as it will otherwise bias our parameter estimates::

    force = fd_curve.f.data
    distance = fd_curve.d.data
    mask = distance < 2.88
    distance = distance[mask]
    force = force[mask]

Now we can load the data into the model::

    fit_twlc.add_data("Twistable WLC", force, distance)

We could add more datasets in a similar manner, but in this example, we only fit
a single model. Let's load the parameters from our previous fit to use them as
initial guesses for this one::

    fit_twlc << fit_odijk

Fit the model
-------------

Now we are ready to fit the model. Considering that the tWLC model is
expensive to evaluate, this may take a while. This is also why we choose
to enable verbose output::

    >>> fit_twlc.fit(verbose=2)
    >>> plt.show()

       Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality
           0              1         1.2449e+02                                    1.72e+05
           1              2         4.4589e+01      7.99e+01       1.39e+01       1.03e+04
           2              3         4.3696e+01      8.93e-01       5.94e+01       1.19e+04
           3              7         4.3302e+01      3.94e-01       4.70e+00       6.55e+02
           4              9         4.3277e+01      2.50e-02       3.47e-01       6.51e+01
           5             11         4.3273e+01      3.68e-03       1.55e+00       7.26e+00
           6             12         4.3268e+01      5.14e-03       3.90e+00       7.58e+00
           7             14         4.3267e+01      7.83e-04       2.03e+00       3.33e+01
           8             15         4.3266e+01      1.76e-03       2.81e-01       1.77e+01
           9             16         4.3264e+01      1.20e-03       2.24e+00       8.83e+00
          10             17         4.3264e+01      8.23e-04       3.65e-01       1.30e+01
          11             19         4.3263e+01      5.01e-04       4.46e-01       1.29e+01
          12             20         4.3263e+01      3.99e-04       5.58e-01       1.78e+00
          13             21         4.3262e+01      7.64e-04       9.83e-01       2.93e+00
          14             22         4.3261e+01      9.86e-04       1.69e+00       4.14e+00
          15             25         4.3261e+01      2.01e-04       2.17e-01       6.69e+00
          16             26         4.3261e+01      2.38e-04       5.50e-01       4.13e+00
          17             27         4.3260e+01      3.69e-04       7.38e-01       2.59e+00
          18             29         4.3260e+01      1.23e-04       4.84e-01       9.65e+00
          19             30         4.3260e+01      1.29e-04       9.80e-02       1.93e+00
          20             31         4.3260e+01      1.54e-04       5.71e-01       1.25e+00
          21             32         4.3260e+01      1.25e-04       5.78e-01       2.65e+00
          22             34         4.3260e+01      1.20e-04       1.71e-01       7.78e+00
          23             35         4.3259e+01      6.24e-05       2.83e-01       8.24e-01
          24             36         4.3259e+01      9.35e-05       4.01e-01       1.23e+00
          25             38         4.3259e+01      2.76e-05       2.46e-01       4.57e+00
          26             39         4.3259e+01      3.12e-05       4.48e-02       9.49e-01
          27             40         4.3259e+01      3.75e-05       2.88e-01       8.47e-01
          28             41         4.3259e+01      1.89e-05       2.58e-01       1.45e+00
          29             43         4.3259e+01      3.46e-05       8.03e-02       4.25e+00
          30             44         4.3259e+01      1.31e-05       1.37e-01       1.46e+00
          31             45         4.3259e+01      1.37e-05       9.26e-02       4.45e-01
          32             46         4.3259e+01      2.50e-05       2.49e-01       5.26e-01
          33             48         4.3259e+01      1.07e-05       1.38e-01       2.73e-01
          34             49         4.3259e+01      8.93e-06       1.77e-01       6.35e-01
          35             51         4.3259e+01      6.56e-06       4.25e-02       2.06e+00
          36             52         4.3259e+01      3.74e-06       7.08e-02       1.45e-01
          37             53         4.3259e+01      5.39e-06       1.06e-01       2.95e-01
          38             55         4.3259e+01      1.08e-06       5.98e-02       1.31e+00
          39             56         4.3259e+01      2.02e-06       8.46e-03       2.34e-01
          40             57         4.3259e+01      2.20e-06       7.18e-02       1.35e-01
          41             58         4.3259e+01      4.54e-07       6.31e-02       3.62e-01
          42             59         4.3259e+01      2.33e-06       1.86e-02       1.10e+00
          43             60         4.3259e+01      5.81e-07       3.14e-02       5.86e-01
          44             61         4.3259e+01      7.49e-07       1.81e-02       1.13e-01
          45             62         4.3259e+01      1.37e-06       6.33e-02       1.24e-01
          46             64         4.3259e+01      5.69e-07       3.30e-02       6.33e-02
          47             65         4.3259e+01      4.70e-07       4.82e-02       1.45e-01
          48             67         4.3259e+01      3.41e-07       1.01e-02       5.26e-01
    `ftol` termination condition is satisfied.
    Function evaluations 67, initial cost 1.2449e+02, final cost 4.3259e+01, first-order optimality 5.26e-01.

Plotting the results
--------------------

After fitting we can plot our results and print our parameters. Doing this
is as simple as invoking `fit.plot()` and `fit.parameters`::

    fit_twlc.plot()
    plt.xlabel('Distance [$\\mu$m]')
    plt.ylabel('Force [pN]');


.. image:: output_9_2.png

We can also show the parameters::

    >>> fit_twlc.parameters

    Name                 Value  Unit        Fitted      Lower bound    Upper bound
    ------------  ------------  ----------  --------  -------------  -------------
    DNA/d_offset     0.145929   NA          True             -inf            inf
    DNA/Lp          40.7095     [nm]        True                0            100
    DNA/Lc           2.64641    [micron]    True                0            inf
    DNA/St        1575.78       [pN]        True                0            inf
    DNA/C          429.285      [pN*nm**2]  True                0           5000
    DNA/g0        -642.876      [pN*nm]     True            -5000              0
    DNA/g1          17.946      [nm]        True                0           1000
    DNA/Fc          35.8221     [pN]        True                0             50
    kT               4.11       [pN*nm]     False               0              8
    DNA/f_offset     0.0497689  [pN]        True               -0.1            0.1

These seem to agree well with what's typically found for dsDNA. Persistence length
around 50, stiffness of about 1600 and g0 and g1 seem to agree well with values
published in literature. Including more data would allow us to increase the precision
and accuracy of our estimates.
