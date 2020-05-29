.. warning::
    This is beta functionality. While usable, this has not yet been tested in a large
    number of different scenarios. The API may also still be subject to change.

Twistable Worm-Like-Chain Fitting
=================================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

First we import the necessary libraries::

    import matplotlib.pyplot as plt
    from lumicks import pylake

Let's load and plot the data first::

    file = lk.File("twlc_data//20200430-163932 FD Curve FD_1_control_forw.h5")
    fd_curve = file.fdcurves["FD_1_control_forw"]
    fd_curve.plot_scatter()

.. image:: output_9_1.png

Set up a basic model first
--------------------------

We clearly see that the force starts levelling out at high forces in the data. This
is a clear sign that to be able to describe this data, we'll need something more complex
in order to capture this behavior. The twistable worm-like chain is one such model that
can describe the untwisting of DNA at high forces. However, its complexity also incurs
some challenges.

Parameter optimization always begins from an initial guess, and if this initial guess
is bad, it can get stuck at an estimated set of parameters that are suboptimal, a
so-called local optimum. One way to mitigate this, is to start at good initial values.

In this notebook, we fit the low part of the curve first, with a regular worm like
chain model, and then use those estimates as initial guesses to fit the twistable model.

Let's set up the Odijk model and create the fit::

    m_odijk = lk.inverted_odijk("DNA").subtract_independent_offset() + lk.force_offset("DNA")
    fit_odijk = lk.FdFit(m_odijk)

Considering that this model only describes the lower part of the curve, we have to
extract the data that is relevant to us. We can obtain this data from the ``FdCurve``
as follows::

    force = fd_curve.f.data
    distance = fd_curve.d.data

We only wish to use the forces below 30, so we filter the data according to this
requirement::

    mask = force < 30
    distance = distance[mask]
    force = force[mask]

Now we are ready to add this data to the fit. Note that it is important to constrain the distance offset, as this
provides a lot of additional freedom in the model::

    fit_odijk.add_data("Inverted Odijk", force, distance)
    fit_odijk["DNA/d_offset"].upper_bound = 0.01
    fit_odijk["DNA/d_offset"].lower_bound = -0.01

And fit the model::

    >>> fit_odijk.fit()

    Fit
      - Model: DNA(x-d)_with_DNA
      - Equation:
          f(d) = argmin[f](norm(DNA.Lc * (1 - (1/2)*sqrt(kT/(f*DNA.Lp)) + f/DNA.St)-(d - DNA.d_offset))) + DNA.f_offset

      - Data sets:
        - FitData(Inverted Odijk, N=959)

      - Fitted parameters:
        Name                  Value  Unit      Fitted      Lower bound    Upper bound
        ------------  -------------  --------  --------  -------------  -------------
        DNA/d_offset    -0.00601252  [au]      True              -0.01           0.01
        DNA/Lp          44.2558      [nm]      True               0            100
        DNA/Lc           2.80085     [micron]  True               0            inf
        DNA/St        1714.56        [pN]      True               0            inf
        kT               4.11        [pN*nm]   False              0              8
        DNA/f_offset     0.0392458   [pN]      True              -0.1            0.1

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
offsets that may exist in the data::

    m_dna = lk.inverted_twistable_wlc("DNA").subtract_independent_offset() + lk.force_offset("DNA")
    fit_twlc = lk.FdFit(m_dna)

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

We could add more datasets in a similar manner, but in this example, we only fit a single model. Let’s load the
parameters from our previous fit to use them as initial guesses for this one. We also fix the twist rigidity and
critical force to values from literature (analogous to Broekmans et al. "DNA twist stability changes with
magnesium (2+) concentration." Physical Review Letters 116, 258102 (2016))::

    fit_twlc.update_params(fit_odijk)

    # Fix twist rigidity and critical force to literature values.
    fit_twlc["DNA/C"].value = 440
    fit_twlc["DNA/C"].fixed = True
    fit_twlc["DNA/Fc"].value = 30.6
    fit_twlc["DNA/Fc"].fixed = True

Fit the model
-------------

Now we are ready to fit the model. Considering that the tWLC model is
expensive to evaluate, this may take a while. This is also why we choose
to enable verbose output::

    >>> fit_twlc.fit(verbose=2)
    >>> plt.show()

       Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality
           0              1         2.4384e+02                                    2.81e+05
           1              2         4.4649e+01      1.99e+02       6.84e+00       1.14e+04
           2              3         4.3820e+01      8.29e-01       5.79e+01       4.67e+03
           3              4         4.3756e+01      6.46e-02       1.36e+01       2.16e+02
           4              5         4.3755e+01      8.30e-04       3.92e+00       9.48e+00
           5              6         4.3755e+01      1.29e-06       7.15e-02       5.84e-02
           6              7         4.3755e+01      5.81e-09       3.60e-02       1.86e-02
    `ftol` termination condition is satisfied.
    Function evaluations 7, initial cost 2.4384e+02, final cost 4.3755e+01, first-order optimality 1.86e-02.

Plotting the results
--------------------

After fitting we can plot our results and print our parameters. Doing this
is as simple as invoking `fit.plot()` and `fit.params`::

    fit_twlc.plot()
    plt.xlabel("Distance [$\\mu$m]")
    plt.ylabel("Force [pN]");


.. image:: output_9_2.png

We can also show the parameters::

    >>> fit_twlc.params

    Name                  Value  Unit        Fitted      Lower bound    Upper bound
    ------------  -------------  ----------  --------  -------------  -------------
    DNA/d_offset    -0.00605829  [au]        True              -0.01           0.01
    DNA/Lp          43.2315      [nm]        True               0            100
    DNA/Lc           2.80289     [micron]    True               0            inf
    DNA/St        1761.79        [pN]        True               0            inf
    DNA/C          440           [pN*nm**2]  False              0           5000
    DNA/g0        -579.909       [pN*nm]     True           -5000              0
    DNA/g1          17.6625      [nm]        True               0           1000
    DNA/Fc          30.6         [pN]        False              0             50
    kT               4.11        [pN*nm]     False              0              8
    DNA/f_offset     0.0295708   [pN]        True              -0.1            0.1

These seem to agree well with what’s typically found for dsDNA.


