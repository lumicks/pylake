.. warning::
    This is beta functionality. While usable, this is a beta-functionality which
    has not yet been tested in a sufficient number of different scenarios. The API
    may also still be subject to change.

DNA FD Fitting
==============

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`


DNA Fd Fitting
--------------

Let's first load our data and see which curves are present in these files::

    >>> control_file = pylake.File("RecA/20200430-192424 FD Curve FD_5_control_forw.h5")
    >>> reca_file = pylake.File("RecA/20200430-192432 FD Curve FD_5_3_RecA_forw.h5")

    >>> print(control_file.fdcurves)
    >>> print(reca_file.fdcurves)

    {'FD_5_control_forw': <lumicks.pylake.fdcurve.FDCurve object at 0x0000022C8514E780>}
    {'FD_5_3_RecA_forw_after_2_quick_manual_FD': <lumicks.pylake.fdcurve.FDCurve object at 0x0000022C8514E860>}

Plot the data
-------------

Alright, we see that each of the files we exported from Bluelake has just one
Fd curve. We can access Fd curves from the file by invoking `control_file[curve_name]`,
or alternatively, since there's only one, we can simply use `popitem`. Let's have a
quick look at the data::

    control_name, control_curve = control_file.fdcurves.popitem()
    reca_name, reca_curve = reca_file.fdcurves.popitem()

    control_curve.plot_scatter(s=1, c='k')
    reca_curve.plot_scatter(s=1, c='r')

.. image:: output_10_1.png

Set up the model
----------------

For this we want to use an inverted WLC model with a force offset.

While it is possible (and equivalent) to generate this model via `pylake.odijk('DNA').invert()`,
an optimized inverted WLC model is explicitly included as a separate model. When only a single WLC
is needed, it is best to use this one, as it is considerably faster than explicitly inverting::

    m_dna = pylake.inverted_odijk("DNA") + pylake.force_offset("DNA")

We would like to fit this model to some data. So let's make a `FdFit`::

    fit = pylake.FdFit(m_dna)

Let's have a look at the parameters in this model::

    >>> print(m_dna.parameter_names)

    ['DNA/Lp', 'DNA/Lc', 'DNA/St', 'kT', 'DNA/f_offset']

Load the data
-------------

Next, we load the data into the model. We have to be careful however, as the Odijk model is only
valid for intermediate forces. That means we'll have to filter our data, to make sure that we remove
any data that's outside the model's valid range. We can do this by creating a logical mask which is
true for the data we wish to include and false for the data we wish to exclude.

The data in an `FdCurve` can be referenced by invoking the f and d attribute for force and distance
respectively. This returns `Slice` objects, from which we can extract the data by calling `.data` on
them. Since we have to do this twice, let's make a little function that extracts the data from the
`FdCurve` and filters it::

    def extract_data(fdcurve, f_min, f_max):
        f = fdcurve.f.data
        d = fdcurve.d.data
        mask = (f < f_max) & (f > f_min)
        return f[mask], d[mask]

    force_control, distance_control = extract_data(control_curve, 0, 25)
    force_reca, distance_reca = extract_data(reca_curve, 0, 25)

We can load data into the `FdFit` by using the function `add_data`. Note that we apply the mask as we
are loading the data using the square brackets::

    data1 = fit.add_data("Control", force_control, distance_control)

If parameters are expected to differ between conditions, we can rename them for a specific data set
when adding data to the fit. For the second data set, we expect the contour length and persistence
length to be different, so let’s rename these. We can do this by passing an extra argument named
`params`. This argument takes a dictionary. The keys of this dictionary have to be given by the
old name of the parameter in the model. This name is typically given by the name of the model
followed by a slash and then the model parameter name. The value of this dictionary should be set
to the model name slash the new parameter name. Let's rename the contour length Lc and persistence
length Lp for this data set::

    data2 = fit.add_data("RecA", force_reca, distance_reca, params={"DNA/Lc": "DNA/Lc_RecA", "DNA/Lp": "DNA/Lp_RecA"})

Set up the fit
--------------

Let's add some custom parameter bounds::

    fit["DNA/Lp"].value = 50
    fit["DNA/Lp"].lower_bound = 39
    fit["DNA/Lp"].upper_bound = 80

    fit["DNA/St"].value = 1200
    fit["DNA/St"].lower_bound = 700
    fit["DNA/St"].upper_bound = 2000

Fit the model
-------------

Everything is set up now. All that remains is to do the fit::

    >>> fit.fit()

    Fit
      - Model: DNA_with_DNA
      - Equation:
          f(d) = argmin[f](norm(DNA.Lc * (1 - (1/2)*sqrt(kT/(f*DNA.Lp)) + f/DNA.St)-d)) + DNA.f_offset

      - Data sets:
        - FitData(Control, N=853)
        - FitData(RecA, N=987, Transformations: DNA/Lp → DNA/Lp_RecA, DNA/Lc → DNA/Lc_RecA)

      - Fitted parameters:
        Name                 Value  Unit      Fitted      Lower bound    Upper bound
        ------------  ------------  --------  --------  -------------  -------------
        DNA/Lp          62.4178     [nm]      True               39             80
        DNA/Lc           2.74467    [micron]  True                0            inf
        DNA/St        1085.61       [pN]      True              700           2000
        kT               4.11       [pN*nm]   False               0              8
        DNA/f_offset     0.0519907  [pN]      True               -0.1            0.1
        DNA/Lp_RecA     63.8319     [nm]      True                0            100
        DNA/Lc_RecA      2.99842    [micron]  True                0            inf


Plot the fit
------------

Plotting the fit alongside the data is easy. Simply call the plot function on the `FdFit` (i.e. `fit.plot()`)::

    fit.plot()
    plt.ylabel('Force [pN]')
    plt.xlabel('Distance [$\\mu$M]')

.. image:: output_10_2.png

We would like to compare the two modelled curves without the data. Plotting these is easy. We can tell the
model to plot the model for a specific data set by slicing the parameters from our fit with the appropriate
data handle: `fit[data1]`. This slice procedure collects exactly those parameters needed to simulate that
condition. The second argument contains the values for the independent variable that we wish to simulate for::

    m_dna.plot(fit[data1], np.arange(2.1, 5.0, .01), 'r--')
    m_dna.plot(fit[data2], np.arange(2.1, 5.0, .01), 'r--')
    plt.ylabel('Force [pN]')
    plt.xlabel('Distance [$\\mu$M]')
    plt.ylim([0, 30])
    plt.xlim([2, 3.1])

.. image:: output_10_3.png

Let’s print the contour length difference due to RecA. We multiply by 1000 since we desire this value in
nanometers::

    >>> print(f"Contour length difference: {(fit['DNA/Lc_RecA'].value - fit['DNA/Lc'].value) * 1000:.2f} [nm]")

    Contour length difference: 253.74 [nm]

Try another model
-----------------

There are more models in pylake. We can also try the Marko Siggia model for instance and see if that fits this
data any differently. Let's fit the Marko Siggia model::

    marko_siggia_fit = pylake.FdFit(pylake.marko_siggia_ewlc_force("DNA").subtract_independent_offset() + pylake.force_offset("DNA"))
    marko_siggia_fit.add_data("Control", force_control, distance_control)
    marko_siggia_fit.add_data("RecA", force_reca, distance_reca, params={"DNA/Lc": "DNA/Lc_RecA", "DNA/Lp": "DNA/Lp_RecA"})
    marko_siggia_fit.fit();

Plot the competing models
-------------------------

Let's plot the models side by side, so we can get an idea of which model fits best::

    plt.figure(figsize=(20,5))
    plt.subplot(1, 2, 1)
    fit.plot()
    plt.title('Odijk')
    plt.ylim([0,10])
    plt.subplot(1, 2, 2)
    marko_siggia_fit.plot()
    plt.title('Marko-Siggia')
    plt.ylim([0,10])

.. image:: output_10_5.png

At first glance, the model fits look very similar. Since we were interested in the contour length
changes, let's have a look at what these models predict for the change in contour length::

    >>> print(f"Contour length difference Odijk: {(fit['DNA/Lc_RecA'].value - fit['DNA/Lc'].value) * 1000:.2f} [nm]")
    >>> print(f"Contour length difference Marko-Siggia: {(marko_siggia_fit['DNA/Lc_RecA'].value - marko_siggia_fit['DNA/Lc'].value) * 1000:.2f} [nm]")

    Contour length difference Odijk: 253.74 [nm]
    Contour length difference Marko-Siggia: 253.68 [nm]

These results are very similar, increasing our confidence in the result.

Which fit is statistically optimal
----------------------------------

We can also determine how well a model fits the data by looking at the corrected Akaike Information
Criterion and Bayesian Information Criterion. Here, a low value indicates a better model.

We can see here that both criteria seem to indicate that the Odijk model provides the best fit.
Please note however, that it is always important to verify that the model produce sensible results.
More freedom to fit parameters, will almost always lead to an improved fit, and this additional
freedom can lead to fits that produce non-physical results. Information criteria tend to try and
penalize unnecessary over-fitting, but they do not guard against unphysical parameter values.

Generally it is always a good idea to try multiple models, and multiple sets of bound constraints,
to get a feel for how reliable the estimates are::

    >>> print("Corrected Akaike Information Criterion")
    >>> print(f"Odijk Model with force offset {fit.aicc}")
    >>> print(f"Marko-Siggia Model with force offset {marko_siggia_fit.aicc}")
    >>> print("Bayesian Information Criterion")
    >>> print(f"Odijk Model with force offset {fit.bic}")
    >>> print(f"Marko-Siggia Model with force offset {marko_siggia_fit.bic}")

    Corrected Akaike Information Criterion
    Odijk Model with single force offset -99.24011262687394
    Marko-Siggia Model with single force offsets -93.61329520140747
    Bayesian Information Criterion
    Odijk Model with single force offset -66.18081403716738
    Marko-Siggia Model with single force offsets -60.5539966117009

We can also quickly compare parameter values::

    >>> fit.parameters

    Name                 Value  Unit      Fitted      Lower bound    Upper bound
    ------------  ------------  --------  --------  -------------  -------------
    DNA/Lp          62.4178     [nm]      True               39             80
    DNA/Lc           2.74467    [micron]  True                0            inf
    DNA/St        1085.61       [pN]      True              700           2000
    kT               4.11       [pN*nm]   False               0              8
    DNA/f_offset     0.0519907  [pN]      True               -0.1            0.1
    DNA/Lp_RecA     63.8319     [nm]      True                0            100
    DNA/Lc_RecA      2.99842    [micron]  True                0            inf

    >>> marko_siggia_fit.parameters

    Name                 Value  Unit      Fitted      Lower bound    Upper bound
    ------------  ------------  --------  --------  -------------  -------------
    DNA/Lp          63.4333     [nm]      True                0            100
    DNA/Lc           2.74361    [micron]  True                0            inf
    DNA/St        1077.36       [pN]      True                0            inf
    kT               4.11       [pN*nm]   False               0              8
    DNA/f_offset     0.0270028  [pN]      True               -0.1            0.1
    DNA/Lp_RecA     64.8165     [nm]      True                0            100
    DNA/Lc_RecA      2.9973     [micron]  True                0            inf

Dynamic experiments
-------------------

We can see some differences in the estimates, but nothing that would be cause for immediate concern,
so let's stick with the Odijk model for the rest of this analysis. One thing we noticed when acquiring
the data was that some of the experiments showed some dynamics. It'd be interesting to look at the
contour length changes for these experiments. To this end, we take the model we just fitted and determine
a contour length per data point of this model, while keeping all other parameters the same.

Let's load the data and have a look::

    dynamic_file = pylake.File("RecA/20200430-182304 FD Curve 40.h5")
    dynamic_name, dynamic_curve = dynamic_file.fdcurves.popitem()
    dynamic_curve.plot_scatter()

.. image:: output_10_6.png

Once again, we extract our data up to 25 pN. We can reuse the function we defined earlier::

    force_dynamic, distance_dynamic = extract_data(dynamic_curve, 0, 25)

A contour length per point
--------------------------

Now comes the more challenging part. Inverting the model for contour length. Luckily, this
procedure has already been implemented in Pylake. The function `parameter_trace` inverts the
model for a particular model parameter. Let's have a look at the parameters it needs::

    >>> help(pylake.parameter_trace)

    Help on function parameter_trace in module lumicks.pylake.fitting.detail.parameter_trace:

    parameter_trace(model, parameters, inverted_parameter, independent, dependent, **kwargs)
        Invert a model with respect to one parameter. This function fits a unique parameter for every data point in
        this data-set while keeping all other parameters fixed. This can be used to for example invert the model with
        respect to the contour length or some other parameter.

        Parameters
        ----------
        model : Model
            Fitting model.
        parameters : Parameters
            Model parameters.
        inverted_parameter : str
            Parameter to invert.
        independent : array_like
            vector of values for the independent variable
        dependent: array_like
            vector of values for the dependent variable
        **kwargs:
            forwarded to scipy.optimize.least_squares

        Examples
        --------
        ::
            # Define the model to be fitted
            model = pylake.inverted_odijk("model") + pylake.force_offset("model")

            # Fit the overall model first
            data_handle = model.add_data("dataset1", f=force_data, d=distance_data)
            current_fit = pylake.Fit(model)
            current_fit.fit()

            # Calculate a per data point contour length
            lcs = parameter_trace(model, current_fit[data_handle], "model/Lc", distance, force)

Let's see if we have all these pieces of information. Our model was called `m_dna`. We can extract
the parameters for the RecA condition using the name we provided to the dataset before (i.e. `fit["RecA"]`).
The parameter we wish to invert for is `DNA/Lc` and for the independent and dependent variables we simply
pass the dataset::

    Lcs = pylake.parameter_trace(m_dna, fit["RecA"], 'DNA/Lc', distance_dynamic, force_dynamic)

Let's plot it::

    plt.plot(Lcs)
    plt.ylabel('Contour lengths')
    plt.xlabel('Time [s]')

.. image:: output_10_7.png

Looks like some of the estimates are way off early in the curve. Doing this inversion at very low
distances is quite error prone, likely due to the non-linearity of the model. In addition, the Odijk
model is known to not be reliable at low forces, so we would like to exclude this data anyway. Let's
only look at the points where the distance is higher than 2.25::

    distance_mask = distance_dynamic > 2.2

    plt.plot(distance_dynamic[distance_mask], Lcs[distance_mask])
    plt.ylabel('Contour length [micron]')
    plt.xlabel('Distance [micron]')

.. image:: output_10_8.png

Here we can see the different contour length transitions quite clearly. There seems to be one region
of contour lengths around 3.2 before finally lengthening to 3.4 micrometers.
