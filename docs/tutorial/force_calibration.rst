Comparing different types of calibration
----------------------------------------

Consider the active calibration from the last section.
This entire calibration can also be performed using only a single function call.
For convenience, assign most of the parameter to a dictionary first::

    shared_parameters = {
        "force_voltage_data": volts.data,
        "bead_diameter": bead_diameter,
        "temperature": 25,
        "sample_rate": volts.sample_rate,
        "driving_data": driving_data.data,
        "driving_frequency_guess": 37,
        "hydrodynamically_correct": False,
    }

Next, unpack this dictionary using the unpacking operator `**`::

    >>> fit = lk.calibrate_force(
    ...     **shared_parameters, active_calibration=True, distance_to_surface=distance_to_surface
    ... )
    >>> print(fit["kappa"].value)
    0.11662183772410809

And compare this to the passive calibration result::

    >>> fit = lk.calibrate_force(
    ...     **shared_parameters, active_calibration=False, distance_to_surface=distance_to_surface
    ... )
    >>> print(fit["kappa"].value)
    0.11763849764570819

These values are quite close.
However, if we do not provide the height above the surface, we can see that the passive calibration result suffers much more than the active calibration result (as passive calibration fully relies on a drag coefficient calculated from the physical input parameters)::

    >>> print(lk.calibrate_force(**shared_parameters, active_calibration=False)["kappa"].value)
    >>> print(lk.calibrate_force(**shared_parameters, active_calibration=True)["kappa"].value)
    0.08616565751377737
    0.11662183772410809

.. note::

    When fitting with the hydrodynamically correct model, the `distance_to_surface` parameter impacts the expected shape of the power spectrum.
    Consequently, when this model is selected, this parameter affects both passive and active calibration.
    For more information on this see the :doc:`theory section on force calibration</theory/force_calibration/force_calibration>` section.

Fast Sensors
------------

Fast detectors have the ability to respond much faster to incoming light resulting in no visible filtering effect in the frequency range we are fitting.
This means that for a fast detector, we do not need to include a filtering effect in our model.
Note that whether you have a fast or slow detector depends on the particular hardware in the C-Trap.
We can omit this effect by passing `fast_sensor=True` to the calibration models or to :func:`~lumicks.pylake.calibrate_force()`.
Note however, that this makes using the hydrodynamically correct model critical, as the simple model doesn't actually capture the data very well.
The following example data acquired on a fast sensor will illustrate why::

    f = lk.File("test_data/fast_measurement_25.h5")

    shared_parameters = {
        "force_voltage_data": decalibrate(f.force2y).data,
        "bead_diameter": 4.38,
        "temperature": 25,
        "sample_rate": volts.sample_rate,
        "fit_range": (1e2, 23e3),
        "num_points_per_block": 200,
        "excluded_ranges": ([190, 210], [13600, 14600])
    }

    plt.figure(figsize=(13, 4))
    plt.subplot(1, 3, 1)
    fit = lk.calibrate_force(**shared_parameters, hydrodynamically_correct=False, fast_sensor=False)
    fit.plot()
    plt.title(f"Simple model + Slow (kappa={fit['kappa'].value:.2f})")
    plt.subplot(1, 3, 2)
    fit = lk.calibrate_force(**shared_parameters, hydrodynamically_correct=False, fast_sensor=True)
    fit.plot()
    plt.title(f"Simple model + Fast (kappa={fit['kappa'].value:.2f})")
    plt.subplot(1, 3, 3)
    fit = lk.calibrate_force(**shared_parameters, hydrodynamically_correct=True, fast_sensor=True)
    fit.plot()
    plt.title(f"Hydrodynamically correct + Fast (kappa={fit['kappa'].value:.2f})")
    plt.tight_layout()
    plt.show()

.. image:: figures/force_calibration/fast_sensors.png

Note how the power spectral fit with the simple model seems to fit the data quite well as long as we also include the filtering effect.
However, the apparent quality of a fit can be deceiving.
Considering that this dataset was acquired on a fast sensor, we should omit the filtering effect.
When the `fast_sensor` flag is enabled, it can be seen that the simple model doesn't actually describe the data.
Switching to the hydrodynamically correct model results in a superior fit to the power spectrum.

So what is happening here? Why did the first fit look good?
When we fit the power spectrum with the simple model and include the filtering effect, the fitting procedure uses the parameters that characterize the filter to fit some of the high frequency attenuation.
With the filtering effect disabled, we obtain a very biased fit because the model fails to fit the data.

If we compare the different fits, we can see that the simple model with filtering effect (`fast_sensor=False`) gives similar stiffness estimates as the hydrodynamically correct model without the filtering.
While this is true for this particular dataset, no general statement can be made about the bias caused by fitting the simple model rather than the hydrodynamically correct power spectrum.
If low bias is desired, one should always use the hydrodynamically correct model when possible.
On regular sensors, it is best to fit the hydrodynamically correct model with the filtering effect enabled.
