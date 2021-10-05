.. warning::
    This is alpha functionality.
    While usable, this has not yet been tested in a large number of different scenarios.
    The API is still be subject to change *without any prior deprecation notice*! If you use this
    functionality keep a close eye on the changelog for any changes that may affect your analysis.

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

Additional parameters can be specified to change the amount of downsampling applied (`num_points_per_block`), the range over which to compute the spectrum (`fit_range`) or to exclude specific frequency ranges from the spectrum (`excluded_ranges`)::

    power_spectrum = lk.calculate_power_spectrum(volts, sample_rate=force_slice.sample_rate, fit_range=(1e2, 23e3), num_points_per_block=2000, excluded_ranges=[(700, 800), (14500, 14600)])

To fit the passive calibration data, we will use a model based on a number of publications by the Flyvbjerg group :cite:`berg2004power,tolic2004matlab,hansen2006tweezercalib,berg2006power`.
This model can be found in :class:`~.PassiveCalibrationModel`. It is calibrated by fitting the following equation to the power spectrum:

.. math::

    P(f) = \frac{D}{\pi ^ 2 \left(f^2 + f_c ^ 2\right)} g(f, f_{diode}, \alpha)

where :math:`D` corresponds to the diffusion constant, :math:`f` the frequency and :math:`f_c` the corner frequency.
The first part of the equation is known as the Lorentzian model and represents the part of the spectrum that originates from the physical motion of the bead.
The second term :math:`g` takes into account the slower response of the position detection system and is given by:

.. math::

    g(f, f_{diode}, \alpha) = \alpha^2 + \frac{1 - \alpha ^ 2}{1 + (f / f_{diode})^2}

Here :math:`\alpha` corresponds to the fraction of the signal response that is instantaneous, while :math:`f_{diode}` characterizes the frequency response of the diode. Note that not all sensor types require this second term.

To convert the parameters obtained from this spectral fit to a trap stiffness, the following is computed:

.. math::

    \kappa = 2 \pi \gamma_0 f_c

Here :math:`\kappa` then represents the final trap stiffness.

We can calibrate the position by considering the diffusion of the bead in squared microns per second:

.. math::

    D = \frac{k_B T}{\gamma_0}

Here :math:`k_B` refers to the Boltzmann constant and :math:`T` reflects the temperature in Kelvin. Comparing this to its measured counterpart in volts squared per second provides us with the desired calibration factor:

.. math::

    \beta = \sqrt{\frac{D_{physical}}{D_{volts}}}

Both of these quantities depend on the parameter :math:`\gamma_0`, which corresponds to the friction coefficient of a sphere and is given by:

.. math::

    \gamma_0 = 3 \pi \eta d

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

These parameters can be accessed as follows::

    >>> print(calibration["kappa"].value)
    >>> print(f.force1x.calibration[1]["kappa (pN/nm)"])
    0.17432391259341345
    0.17431947810792106

We can plot the calibration by calling::

    calibration.plot()

.. image:: force_calibration_fit.png

Note that by default, a bias correction is applied to the fitted results :cite:`norrelykke2010power`.
This bias correction is applied to the diffusion constant and amounts to a correction of :math:`\frac{N}{N+1}`, where :math:`N` refers to the number of points used for a particular spectral data point.
It can optionally be disabled by passing `bias_correction=False` to :func:`~lumicks.pylake.fit_power_spectrum`.

Hydrodynamically correct model
------------------------------

While the idealized Lorentzian model discussed in the previous section is often sufficiently accurate, there are scenarios where more detailed models are necessary.

The idealized model is based on the assumption that the drag force is only proportional to the bead's velocity.
This assumption is realistic when the bead moves at a constant speed with respect to the fluid.
When oscillating a bead however, this frictional force becomes frequency dependent.

The following equation accounts for a frequency dependent drag coefficient :cite:`tolic2006calibration`:

.. math::

    P_{hydro}(f) = \frac{D Re(\gamma / \gamma_0)}{\pi^2 \left(\left(f_{c,0} + f Im(\gamma/\gamma_0) - f^2/f_{m, 0}\right)^2 + \left(f Re(\gamma / \gamma_0)\right)^2\right)}

where the corner frequency is given by:

.. math::

    f_{c, 0} = \frac{\kappa}{2 \pi \gamma_0}

and :math:`f_{m, 0}` parameterizes the time it takes for friction to dissipate the kinetic energy of the bead:

.. math::

    f_{m, 0} = \frac{\gamma_0}{2 \pi m}

with :math:`m` the mass of the bead.
Finally, :math:`\gamma` corresponds to the frequency dependent drag coefficient.
For measurements in bulk, far away from a surface, :math:`\gamma` = :math:`\gamma_{stokes}`, where :math:`\gamma_{stokes}` is given by:

.. math::

    \gamma_{stokes} = \gamma_0 \left(1 + (1 - i)\sqrt{\frac{f}{f_{\nu}}} - \frac{2}{9}\frac{f}{f_{\nu}} i\right)

Where :math:`f_{\nu}` is the frequency at which the penetration depth equals the radius of the bead, :math:`4 \nu/(\pi d^2)` with :math:`\nu` the kinematic viscosity.

This approximation is reasonable, when the bead is far from the surface.

When approaching the surface, the drag experienced by the bead depends on the distance between the bead and the surface of the flow cell.
An approximate expression for the frequency dependent drag coefficient is then given by :cite:`tolic2006calibration`:

.. math::

    \gamma(f, R/l) = \frac{\gamma_{stokes}(f)}{1 - \frac{9}{16}\frac{R}{l}\left(1 - \left((1 - i)/3\right)\sqrt{\frac{f}{f_{\nu}}} + \frac{2}{9}\frac{f}{f_{\nu}}i - \frac{4}{3}(1 - e^{-(1-i)(2l-R)/\delta})\right)}

Where :math:`\delta = R \sqrt{\frac{f_{\nu}}{f}}` represents the aforementioned penetration depth, :math:`R` corresponds to the bead radius and :math:`l` to the distance from the bead center to the nearest surface.

While these models may look daunting, they are all available in Pylake and can be used by simply providing a few additional arguments to the :class:`~.PassiveCalibrationModel`.
It is recommended to use these equations when less than 10% systematic error is desired :cite:`tolic2006calibration`.

The figure below shows the difference between the hydrodynamically correct spectrum and the regular Lorentzian for various bead sizes.

.. image:: hydro.png

These more advanced models require a few extra parameters namely the density of the sample, density of the bead and distance to the surface (in meters)::

    force_model = lk.PassiveCalibrationModel(bead_diameter, hydrodynamically_correct=True, rho_sample=999, rho_bead=1060.0, distance_to_surface=1e-6)

Note that when `rho_sample` and `rho_bead` are omitted, values for water and polystyrene are used for the sample and bead density respectively.

Additionally, when the parameter `distance_to_surface` is omitted, a simpler model is used which assumes the experiment was performed deep in bulk (neglecting the increased drag induced by the nearby surface).

Fast Sensors
------------

Up to now, we've always fitted a physical spectrum multiplied by a filtering effect.
This filtering effect arose because a fraction of the light is not measured instantaneously by the detector.
Fast detectors respond much faster resulting in no visible filtering effect by the detector in the frequency range we are fitting, meaning that we do not need to model the slower response time of the diode.
We can omit the diode response model by passing `fast_sensor=True` to the `CalibrationModel`, this removes the diode part from the model entirely::

    force_model = lk.PassiveCalibrationModel(bead_diameter, viscosity=0.001002, temperature=20, fast_sensor=True, hydrodynamically_correct=True)

Note however, that this makes using the hydrodynamically correct model critical, as the Lorentzian doesn't actually capture the data very well.
This is illustrated in the figure below.
Here we see the same power spectrum (acquired on a fast detector) fitted with three different models.

.. image:: fast_hydro.png

Here we can see that the fit with the Lorentzian model with the diode filtering effect seems to fit the data quite well.
As we can see in the comparisons above with the hydrodynamics on and off, including hydrodynamics attenuates higher frequencies (an effect similar to a low pass filter).
In the case of the Lorentzian with a diode model, the fitting procedure has used the diode model to fit some of this high frequency attenuation.
However, when we enable the `fast_sensor` flag, we see that the Lorentzian model doesn't actually describe the data.
With the diode model disabled, we obtain a very biased fit.
Enabling the hydrodynamic corrections, we can see that this describes the power spectrum much better.
If we compare the different fits, we can see that the Lorentzian model with diode effect (`fast_sensor=False`) gives similar stiffness estimates as the hydrodynamically correct model without the diode effect.
While this is true for this particular dataset, no general statement can be made about the bias of fitting a Lorentzian rather than the hydrodynamically correct power spectrum.
If low bias is desired, one should use the hydrodynamically correct model.
On regular sensors, it is best to fit the hydrodynamically correct model with the diode model enabled.
