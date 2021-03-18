Force calibration
=================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

Force calibration is used to convert the raw voltages that Bluelake records to actual forces.
In this tutorial, we will be redoing a calibration that we already performed in Bluelake, to provide an example of using the force calibration module::

    f = lk.File("calibration.h5")

In this tutorial, we want to reproduce calibration item 1.
We grab the chunk of data that was used for the calibration last time::

    start = f.force1x.calibration[1]["Start time (ns)"]
    stop = f.force1x.calibration[1]["Stop time (ns)"]
    force_slice = f.force1x[start:stop]

To be able to calibrate our forces, we first convert back our data to raw voltages.
We can do this using the previous calibration performed in Bluelake (in this case, the first one in the file)::

    offset = f.force1x.calibration[0]["Offset (pN)"]
    response = f.force1x.calibration[0]["Rf (pN/V)"]
    volts = (force_slice.data - offset) / response

Force calibration models are fit to power spectra. To compute a power spectrum from our data::

    power_spectrum = lk.calculate_power_spectrum(volts, sample_rate=force_slice.sample_rate)

This function returns a power_spectrum which we can plot::

    power_spectrum.plot()

.. image:: force_calibration_blocked_spectrum.png

Note that the computation of the power spectrum involves some downsampling.
If you wish to change the amount of downsampling applied or which range to compute the spectrum for, you can specify these as additional arguments::

    lk.calculate_power_spectrum(volts, sample_rate=force_slice.sample_rate, fit_range=(1e2, 23e3), num_points_per_block=350)

To fit the passive calibration data, we will use a model based on a number of publications by the Flyvbjerg group :cite:`berg2004power,tolic2004matlab,hansen2006tweezercalib,berg2006power`.
This model can be found in :class:`~.PassiveCalibrationModel`. It is calibrated by fitting the following equation to the power spectrum:

.. math::

    P(f) = \frac{D}{2 \pi ^ 2 \left(f^2 + f_c ^ 2\right)} g(f, f_{diode}, \alpha)

where :math:`D` corresponds to the diffusion constant, :math:`f` the frequency and :math:`f_c` the fitted cutoff. The second term :math:`g` takes into account the slower response of the position detection system and is given by:

.. math::

    g(f, f_{diode}, \alpha) = \alpha^2 + \frac{1 - \alpha ^ 2}{1 + (f / f_{diode})^2}

Here :math:`\alpha` corresponds to the fraction of the signal response that is instantaneous, while :math:`f_{diode}` characterizes the frequency response of the diode.

To convert the parameters obtained from this spectral fit to a trap stiffness, the following is computed:

.. math::

    \kappa = 2 \pi \gamma f_c

Here :math:`\kappa` then represents the final trap stiffness. The parameter :math:`\gamma` corresponds to the friction coefficient of a sphere, which is given by:

.. math::

    \gamma = 3 \pi \eta d

where :math:`\eta` corresponds to the dynamic viscosity and :math:`d` represents the bead diameter.

We use the bead diameter found in the calibration performed in Bluelake.
You can optionally also provide a viscosity (in Pa/s) and temperature (in degrees Celsius)::

    bead_diameter = f.force1x.calibration[1]["Bead diameter (um)"]
    force_model = lk.PassiveCalibrationModel(bead_diameter, viscosity=0.001002, temperature=20)

To fit this model to the data, you can now invoke::

    calibration = lk.fit_power_spectrum(power_spectrum, force_model)
    calibration

This will produce a table with your fitted calibration parameters.

.. image:: force_calibration_table.png

These parameters can be accessed analogously to the calibrations in Bluelake files::

    >>> print(calibration["kappa (pN/nm)"])
    >>> print(f.force1x.calibration[1]["kappa (pN/nm)"])
    0.17432391259341345
    0.17431947810792106

We can plot the calibration by calling::

    calibration.plot()

.. image:: force_calibration_fit.png
