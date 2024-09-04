Getting good calibrations
-------------------------

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

The number of options presented when calling :func:`~lumicks.pylake.calibrate_force()` may be daunting at
first, but hopefully, this chapter will provide some guidelines on how to obtain accurate and
precise force calibrations using Pylake.

Experimental considerations
***************************

Core parameters
"""""""""""""""

Passive also referred to as thermal calibration involves fitting a model to data of a bead jiggling in a trap.
This model relies on a number of parameters that have to be specified in order to get the correct calibration factors.
The most important parameters are:

- The bead diameter (in microns).
- The viscosity of the medium, which strongly depends on :ref:`temperature<temperature_theory>`.
- The distance to the surface (in case of a surface experiment).

To find the viscosity of water at a particular temperature, Pylake uses :func:`~lumicks.pylake.viscosity_of_water` which implements the model presented in :cite:`huber2009new`.
When omitted, this function will automatically be used to look up the viscosity of water for that particular temperature

.. image:: figures/temperature_dependence.png
  :nbattach:

.. note::

    Note that for experiments that use a different medium than water, the viscosity at the experimental temperature should explicitly be provided.

Hydrodynamically correct model
""""""""""""""""""""""""""""""

For lateral calibration, it is recommended to use the
:ref:`hydrodynamically correct theory<hydro_model_theory>` by setting `hydrodynamically_correct` to
`True`) as it provides an improved model of the underlying physics.
For small beads (< 1 micron) the differences will be small, but for larger beads substantial
differences can occur. There is only one exception to this recommendation, which is when the beads
are so close to the flowcell surface (0.75 x diameter) that this model becomes invalid.

.. image:: figures/surface_calibration_workflow.png
  :nbattach:

Using the hydrodynamically correct theory requires a few extra parameters: the density of the
sample (`rho_sample`) and bead (`rho_bead`). When these are not provided,
Pylake uses values for water and polystyrene for the sample and bead density respectively.

Experiments near the surface
""""""""""""""""""""""""""""

So far, we have only considered experiments performed deep in the flow-cell.
In reality, proximity of the flowcell surface to the bead leads to an increase in the drag force on the bead.

When doing experiments near the surface, it is recommended to provide a `distance_to_surface`.
This distance should be the distance from the center of the bead to the surface of the flowcell.
Since it can be challenging to determine this distance, it is recommended to use active calibration
when calibrating near the surface, since this makes calibration far less sensitive to mis-specification
of the bead diameter and height.

Active or passive?
""""""""""""""""""

Active calibration does not rely as much on the bead diameter, viscosity of the medium and distance to the surface as passive calibration does.
For this reason, :ref:`active calibration<active_calibration_theory>` is highly recommended when the bead is close to the surface, as it is less sensitive to the distance to the surface and bead diameter.
It is also recommended when the viscosity of the medium or the bead diameter are poorly known.
In the case of active calibration, it is mandatory to provide a nanostage signal, as well as a guess of the driving frequency.
Let's compare passive and active calibration for a surface experiment::

    f = lk.File("test_data/near_surface_active_calibration.h5")

    previous_calibration = f.force1x.calibration[0]
    bead_diameter = previous_calibration["Bead diameter (um)"]

    # Calibration was performed at 1.04 * bead_diameter
    distance_to_surface = 1.04 * bead_diameter

    # Decalibrate the force data
    volts = f.force1x / previous_calibration.force_sensitivity

    # Grab the nanostage data
    driving_data = f["Nanostage position"]["X"]

    # Since we will be doing a few calibrations, let's store the parameters in a dictionary
    shared_parameters = {
        "force_voltage_data": volts.data,
        "bead_diameter": bead_diameter,
        "temperature": 25,
        "sample_rate": volts.sample_rate,
        "driving_data": driving_data.data,
        "driving_frequency_guess": 37,
        "hydrodynamically_correct": False,  # We will be too close to the surface for this model
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
In this experiment, we accurately determined the distance to the surface, but in most cases, this surface is only known very approximately.
However, if we do not provide the height above the surface, we can see that the passive calibration result suffers much more than the active calibration result (as passive calibration fully relies on a drag coefficient calculated from the physical input parameters)::

    >>> print(lk.calibrate_force(**shared_parameters, active_calibration=False)["kappa"].value)
    >>> print(lk.calibrate_force(**shared_parameters, active_calibration=True)["kappa"].value)
    0.08616565751377737
    0.11662183772410809

.. note::

    Note that the drag coefficient `gamma_ex` that Pylake returns always corresponds to the drag coefficient extrapolated back to its bulk value.
    This ensures that drag coefficients can be compared and carried over between experiments performed at different heights.
    The field `local_drag_coefficient` contains an estimate of the local drag coefficient (at the provided height).

Axial force
"""""""""""

When calibrating axial forces, it is important to set the `axial` flag to `True`.
Note that we currently do not support hydrodynamically correct models for axial calibrations.
Setting `axial` to `True` means you have to set `hydrodynamically_correct` to `False`.

Technical considerations
************************

Sensor
""""""

In addition to the model that describes the bead's motion, it is important to take into account the
:ref:`characteristics of the sensor<diode_theory>` used to measure the data.

A silicon diode sensor is characterized by two parameters, a "relaxation factor" `alpha` and frequency `f_diode`.
These parameters can either be estimated along with the other parameters or characterized independently.

When estimated, care must be taken that the corner frequency of the power spectrum `fc` is
:ref:`lower than the estimated diode frequency<high_corner_freq>`.
You can check whether a calibration item had to estimate parameters from the calibration
data by checking the property `recalibrated.fitted_diode`.
When this property is set to true, it means that the diode parameters were not fixed during the fit.

.. warning::

    For high corner frequencies, calibration can become unreliable when the diode parameters are fitted.
    A warning sign for this is when the corner frequency `fc` approaches or goes beyond the diode frequency `f_diode`.
    For more information see the section on :ref:`High corner frequencies`.

Fit range
"""""""""

The fit range determines which section of the power spectrum is actually fit.
Two things are important when choosing a fitting range:

The corner frequency should be clearly visible in the fitted range (frequency where the slope increases).
When working at low laser laser powers, it is possible that the :ref:`noise floor is visible<noise_floor>` at higher frequencies.
This noise floor should always be excluded from the fit.
Below is an example of a bad fit due to a noise floor.
Note how the spectrum flattens out at high frequencies and the model is unable to capture this.

.. image:: figures/bad_fit_noise_floor.png

As a rule of thumb, an upper bound of approximately four times the corner frequency is usually a safe margin.
The fitting bounds can be specified by providing a `fit_range` to any of the calibration functions.

Frequency exclusion ranges
""""""""""""""""""""""""""

Force calibration is very sensitive to outliers.
It is therefore important to either exclude noise peaks from the data prior to fitting.
Excluding noise floors can be done by providing a list of tuples to the `excluded_ranges` argument of :func:`~lumicks.pylake.calibrate_force()`::

    force_data = lk.File("test_data/robust_fit_data.h5").force2y

    params = {
        "bead_diameter": 4.4,
        "temperature": 25,
        "num_points_per_block": 200,
        "hydrodynamically_correct": True,
    }

    calibration1 = lk.calibrate_force(
        force_data.data, sample_rate=force_data.sample_rate, **params
    )
    calibration2 = lk.calibrate_force(
        force_data.data, sample_rate=force_data.sample_rate, **params, excluded_ranges=[(19447, 19634)]
    )

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1)
    calibration1.plot(show_excluded=True)
    plt.title(f"Stiffness: {calibration1.stiffness:.3f}, Force sensi: {calibration1.displacement_sensitivity:.2f}")
    plt.subplot(2, 2, 2)
    calibration2.plot(show_excluded=True)
    plt.tight_layout()
    plt.title(f"Stiffness: {calibration2.stiffness:.3f}, Force sensi: {calibration2.displacement_sensitivity:.2f}")

    plt.subplot(2, 2, 3)
    calibration1.plot_spectrum_residual()
    plt.ylim([0, 2])
    plt.title(f"Stiffness: {calibration1.stiffness:.3f}, Force sensi: {calibration1.displacement_sensitivity:.2f}")
    plt.subplot(2, 2, 4)
    calibration2.plot_spectrum_residual()
    plt.tight_layout()
    plt.title(f"Stiffness: {calibration2.stiffness:.3f}, Force sensi: {calibration2.displacement_sensitivity:.2f}")

.. image:: figures/frequency_exclusion_ranges.png

Note that when plotting the calibration, we have used `show_excluded=True`, which shows the excluded ranges in the plot.
We can request these excluded ranges from the calibration item itself::

    >>> calibration2.excluded_ranges
    [(19447, 19634)]

This will also work for an item coming from Bluelake.
An alternative to specifying frequency exclusion ranges manually is :ref:`robust fitting<robust_fitting>`.
Robust fitting is less sensitive to outliers and can be used to make the fitting procedure less sensitive.
It also has an option to detect noise peaks and :ref:`determine exclusion ranges automatically<find_fer>`.

Blocking
""""""""





.. list-table:: Calibration Parameters
   :widths: 25 75
   :header-rows: 1

   * - parameter
     - recommendation
   * - bead_diameter
     - Bead diameter in microns
   * - temperature
     - Temperature in Celsius. Note that the temperature may be higher than ambient due to laser heating.
   * - viscosity
     - Can be omitted if water. When provided, remember that this strongly depends on temperature.
   * - hydrodynamically_correct
     - Set to `True` when further away from the surface than 0.75 x diameter.
   * - axial
     - Set to true if this is an axial calibration.
   * - fast_sensor
     - Disables the diode model. Set to `True` when using a fast detector, such as the one used in axial calibration.
   * - fit_range
     - Set such that the noise floor is excluded from the fitting range. Ensure the corner frequency is clearly visible in this range.
   * -

active_calibration : bool, optional
    Active calibration, when set to True, driving_data must also be provided.
driving_data : numpy.ndarray, optional
    Array of driving data.
driving_frequency_guess : float, optional
     Guess of the driving frequency. Required for active calibration.
axial : bool, optional
    Is this an axial calibration? Only valid for a passive calibration.
hydrodynamically_correct : bool, optional
    Enable hydrodynamically correct model.
rho_sample : float, optional
    Density of the sample [kg/m**3]. Only used when using hydrodynamically correct model.
rho_bead : float, optional
    Density of the bead [kg/m**3]. Only used when using hydrodynamically correct model.
distance_to_surface : float, optional
    Distance from bead center to the surface [um]
    When specifying `None`, the model will use an approximation which is only suitable for
    measurements performed deep in bulk.
fast_sensor : bool, optional
     Fast sensor? Fast sensors do not have the diode effect included in the model.
fit_range : tuple of float, optional
    Tuple of two floats (f_min, f_max), indicating the frequency range to use for the full model
    fit. [Hz]
num_points_per_block : int, optional
    The spectrum is first block averaged by this number of points per block.
    Default: 2000.
excluded_ranges : list of tuple of float, optional
    List of ranges to exclude specified as a list of (frequency_min, frequency_max).
drag : float, optional
    Overrides the drag coefficient to this particular value. Note that you want to use the
    bulk drag coefficient for this (obtained from the field `gamma_ex`). This can be used to
    carry over an estimate of the drag coefficient obtained using an active calibration
    procedure.
fixed_diode : float, optional
    Fix diode frequency to a particular frequency.
fixed_alpha : float, optional
    Fix diode relaxation factor to particular value.
