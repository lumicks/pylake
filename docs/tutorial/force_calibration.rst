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
Passive calibration is also often referred to as thermal calibration.
This model can be found in :class:`~.PassiveCalibrationModel`. It is calibrated by fitting the following equation to the power spectrum:

.. math::

    P(f) = \frac{D_\mathrm{measured}}{\pi ^ 2 \left(f^2 + f_c ^ 2\right)} g(f, f_\mathrm{diode}, \alpha) \quad \mathrm{[V^2/Hz]}

where :math:`D_\mathrm{measured}` corresponds to the diffusion constant (in `volts`), :math:`f` the frequency and :math:`f_c` the corner frequency.
The first part of the equation is known as the Lorentzian model and represents the part of the spectrum that originates from the physical motion of the bead.
The second term :math:`g` takes into account the slower response of the position detection system and is given by:

.. math::

    g(f, f_\mathrm{diode}, \alpha) = \alpha^2 + \frac{1 - \alpha ^ 2}{1 + (f / f_\mathrm{diode})^2}

Here :math:`\alpha` corresponds to the fraction of the signal response that is instantaneous, while :math:`f_\mathrm{diode}` characterizes the frequency response of the diode. Note that not all sensor types require this second term.

To convert the parameters obtained from this spectral fit to a trap stiffness, the following is computed:

.. math::

    \kappa = 2 \pi \gamma_0 f_c \quad \mathrm{[N/m]}

Here :math:`\kappa` then represents the final trap stiffness.

We can calibrate the position by considering the diffusion of the bead:

.. math::

    D_\mathrm{physical} = \frac{k_B T}{\gamma_0} \quad \mathrm{[m^2/s]}

Here :math:`k_B` is the Boltzmann constant and :math:`T` is the local temperature in Kelvin. Comparing this to its measured counterpart in volts squared per second provides us with the desired calibration factor:

.. math::

    R_d = \sqrt{\frac{D_\mathrm{physical}}{D_\mathrm{measured}}} \quad \mathrm{[m/V]}

Both of these quantities depend on the parameter :math:`\gamma_0`, which corresponds to the drag coefficient of a sphere and is given by:

.. math::

    \gamma_0 = 3 \pi \eta d \quad \mathrm{[kg/s]}

where :math:`\eta` corresponds to the dynamic viscosity [Pa*s] and :math:`d` is the bead diameter [m].
Note that in `pylake` we actually use `microns` to specify the bead diameter for convenience.

We use the bead diameter found in the calibration performed in Bluelake.
You can optionally also provide a viscosity (in Pa/s) and temperature (in degrees Celsius)::

    bead_diameter = f.force1x.calibration[1]["Bead diameter (um)"]
    force_model = lk.PassiveCalibrationModel(bead_diameter, viscosity=0.001002, temperature=20)

To find the viscosity of water at a particular temperature, you can use :func:`~lumicks.pylake.viscosity_of_water` :cite:`huber2009new`.
When omitted, this function will automatically be used to look up the viscosity of water for that particular temperature.
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

    P_\mathrm{hydro}(f) = \frac{D \mathrm{Re}(\gamma / \gamma_0)}{\pi^2 \left(\left(f_{c,0} + f \mathrm{Im}(\gamma/\gamma_0) - f^2/f_{m, 0}\right)^2 + \left(f \mathrm{Re}(\gamma / \gamma_0)\right)^2\right)}

where the corner frequency is given by:

.. math::

    f_{c, 0} = \frac{\kappa}{2 \pi \gamma_0} \quad \mathrm{[Hz]}

and :math:`f_{m, 0}` parameterizes the time it takes for friction to dissipate the kinetic energy of the bead:

.. math::

    f_{m, 0} = \frac{\gamma_0}{2 \pi m} \quad \mathrm{[Hz]}

with :math:`m` the mass of the bead.
Finally, :math:`\gamma` corresponds to the frequency dependent drag coefficient.
For measurements in bulk, far away from a surface, :math:`\gamma` = :math:`\gamma_\mathrm{stokes}`, where :math:`\gamma_\mathrm{stokes}` is given by:

.. math::

    \gamma_\mathrm{stokes} = \gamma_0 \left(1 + (1 - i)\sqrt{\frac{f}{f_{\nu}}} - \frac{2}{9}\frac{f}{f_{\nu}} i\right) \quad \mathrm{[kg/s]}

Where :math:`f_{\nu}` is the frequency at which the penetration depth equals the radius of the bead, :math:`4 \nu/(\pi d^2)` with :math:`\nu` the kinematic viscosity.

This approximation is reasonable, when the bead is far from the surface.

When approaching the surface, the drag experienced by the bead depends on the distance between the bead and the surface of the flow cell.
An approximate expression for the frequency dependent drag coefficient is then given by :cite:`tolic2006calibration`:

.. math::

    \gamma(f, R/l) = \frac{\gamma_\mathrm{stokes}(f)}{1 - \frac{9}{16}\frac{R}{l}\left(1 - \left((1 - i)/3\right)\sqrt{\frac{f}{f_{\nu}}} + \frac{2}{9}\frac{f}{f_{\nu}}i - \frac{4}{3}(1 - e^{-(1-i)(2l-R)/\delta})\right)} \quad \mathrm{[kg/s]}

Where :math:`\delta = R \sqrt{\frac{f_{\nu}}{f}}` represents the aforementioned penetration depth, :math:`R` corresponds to the bead radius and :math:`l` to the distance from the bead center to the nearest surface.

While these models may look daunting, they are all available in `pylake` and can be used by simply providing a few additional arguments to the :class:`~.PassiveCalibrationModel`.
It is recommended to use these equations when less than 10% systematic error is desired :cite:`tolic2006calibration`.

The figure below shows the difference between the hydrodynamically correct spectrum and the regular Lorentzian for various bead sizes.

.. image:: hydro.png

These more advanced models require a few extra parameters namely the density of the sample, density of the bead and distance to the surface (in meters)::

    force_model = lk.PassiveCalibrationModel(bead_diameter, hydrodynamically_correct=True, rho_sample=999, rho_bead=1060.0, distance_to_surface=1e-6)

Note that when `rho_sample` and `rho_bead` are omitted, values for water and polystyrene are used for the sample and bead density respectively.

Additionally, when the parameter `distance_to_surface` is omitted, a simpler model is used which assumes the experiment was performed deep in bulk (neglecting the increased drag induced by the nearby surface).

Faxen's law
-----------

The hydrodynamically correct model presented in the previous section works well when the bead center is at least 1.5 times the radius above the surface.

When going closer, the drag effect becomes stronger than the frequency dependent effects and better models to approximate the local drag exist.

For lateral calibration, the following approximation is typically used :cite:`schaffer2007surface`:

.. math::

    \gamma_\mathrm{faxen}(R/l) = \frac{\gamma_0}{
        1 - \frac{9R}{16l} + \frac{1R^3}{8l^3} - \frac{45R^4}{256l^4} - \frac{1R^5}{16l^5}
    }

We can use this model by setting `hydrodynamically_correct` to `False`, while still providing a distance to the surface::

    force_model = lk.PassiveCalibrationModel(bead_diameter, hydrodynamically_correct=False, distance_to_surface=1e-6)

Note that `pylake` always returns the bulk drag coefficient :math:`\gamma_0`.

Axial Calibration
-----------------

For calibration in the axial direction, no hydrodynamically correct theory exists.
In this case, one should use a Lorentzian with a specific correction term :cite:`schaffer2007surface`:

.. math::

    \gamma_\mathrm{axial}(R/l) = \frac{\gamma_0}{
        1.0
        - \frac{9R}{8l}
        + \frac{1R^3}{2l^3}
        - \frac{57R^4}{100l^4}
        + \frac{1R^5}{5l^5}
        + \frac{7R^{11}}{200l^{11}}
        - \frac{1R^{12}}{25l^{12}}
    }

This model deviates less than 0.1% from Brenner's exact formula for :math:`l/R >= 1.1` and less than 0.3% over the entire range of :math:`l` :cite:`schaffer2007surface`:.

This model can be used in Pylake by specifying `axial=True`::

    force_model = lk.PassiveCalibrationModel(bead_diameter, distance_to_surface=1e-6, axial=True)

Note that no hydrodynamically correct model is available for axial calibration.

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

Active Calibration
------------------

For certain applications, passive force calibration, as described above, is not sufficiently accurate.
Using active calibration, the accuracy of the calibration can be improved, because active calibration uses fewer assumptions than passive calibration.

When performing passive calibration, we base our calculations on a theoretical drag coefficient.
This theoretical drag coefficient depends on parameters that are only known with limited precision: the diameter of the bead and the viscosity.
This viscosity in turn depends strongly on the local temperature around the bead, which is typically poorly known.

During active calibration, the trap or nanostage is oscillated sinusoidally.
These oscillations result in a driving peak in the force spectrum.
Using power spectral analysis, the force can then be calibrated without prior knowledge of the drag coefficient.

When the power spectrum is computed from an integer number of oscillations, the driving peak is visible at a single data point at :math:`f_\mathrm{drive}`.

.. image:: driving_input.png

The physical spectrum is then given by a thermal part (like before) and an active part:

.. math::

    P^\mathrm{thermal}(f) = \frac{D}{\pi ^ 2 \left(f^2 + f_c ^ 2\right)}

    P^\mathrm{active}(f) = \frac{A^2}{2\left(1 + \frac{f_c^2}{f_\mathrm{drive}^2}\right)} \delta(f - f_\mathrm{drive})

    P^\mathrm{total}(f) = P^\mathrm{thermal}(f) + P^\mathrm{active}(f)

Since we know the driving amplitude, :math:`A`, we know how the bead reacts to the driving motion and we can observe this response in the power spectral density (PSD), we can use this relation to determine the positional calibration.

If we use the basic Lorentzian model, then the theoretical power (integral over the delta spike) corresponding to the driving input is given by :cite:`tolic2006calibration`:

.. math::

    W_\mathrm{physical} = \frac{A^2}{2\left(1 + \frac{f_c^2}{f_\mathrm{drive}^2}\right)} \quad \mathrm{[m^2]}

Subtracting the thermal part of the spectrum, we can determine the same quantity experimentally.

.. math::

    W_\mathrm{measured} = \left(P_\mathrm{measured}^\mathrm{total}(f_\mathrm{drive}) - P_\mathrm{measured}^\mathrm{thermal}(f_\mathrm{drive})\right) \Delta f \quad \mathrm{[V^2]}

:math:`\Delta f` refers to the width of one spectral bin.
Here the thermal contribution that needs to be subtracted is obtained from fitting the thermal part of the spectrum using the passive calibration procedure from before.
The desired positional calibration is then:

.. math::

    R_d = \sqrt{\frac{W_\mathrm{physical}}{W_\mathrm{measured}}} \quad \mathrm{[m/V]}

Note how this time around, we did not rely on assumptions on the viscosity of the medium or the bead size.

As a side effect of this calibration, we actually obtain an experimental estimate of the drag coefficient:

.. math::

    \gamma_\mathrm{measured} = \frac{k_B T}{R_d^2 D_\mathrm{measured}} \quad \mathrm{[kg/s]}

How to do active calibration in Pylake
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using `pylake`, the procedure to use active calibration is not very different from passive calibration.
However, it does require some additional data channels as inputs.
Instead of using the :class:`~.PassiveCalibrationModel` presented in the previous section, we now use the :class:`ActiveCalibrationModel`.

In this tutorial, we're going to assume the nanostage was used as driving input::

    driving_data = f["Nanostage Position"]["Nanostage X"][start:stop]

We also need to provide the sample rate at which the data was acquired, and a rough guess for the driving frequency.
`pylake` will find an accurate estimate of the driving frequency based on this initial estimate (provided that it is close enough)::

    active_model = lk.ActiveCalibrationModel(driving_data.data, volts, driving_data.sample_rate, bead_diameter, driving_frequency_guess=37)

To check the determined frequency, we can look at the determined driving frequency::

    >>> active_model.driving_frequency
    36.95

We can now use this model to fit the power spectrum::

    calibration = lk.fit_power_spectrum(power_spectrum, active_model)
    calibration

And that's all there is to it.

Analogously, we can specify `hydrodynamically_correct=True` if we wish to use the hydrodynamically correct theory here.
This fits the thermal part with the hydrodynamically correct power spectrum and also uses a hydrodynamically correct model for the peak:

.. math::

    P_\mathrm{hydro}^\mathrm{active}(f) = \delta \left(f - f_\mathrm{drive}\right) \frac{\left(A f_\mathrm{drive} \left|\gamma / \gamma_0\right|\right)^2}{2 \left(\left(f_{c,0} + f \mathrm{Im}(\gamma/\gamma_0) - f^2/f_{m, 0}\right)^2 + \left(f \mathrm{Re}(\gamma / \gamma_0)\right)^2\right)}

We can also include a distance to the surface like before.
This results in the expression for `\gamma` becoming dependent on the distance to the surface.
This uses the same expression as listed in the section on the :ref:`hydrodynamically correct model<Hydrodynamically correct model>`.

One thing to note is that when using the hydrodynamically correct model, the equation for the drag _does_ include the viscosity and bead diameter.
However, they now appear in a term which already amounts to a small correction therefore the impact of any errors in these is reduced :cite:`tolic2006calibration`.
