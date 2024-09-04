Calibration items
-----------------

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

This tutorial will focus on performing force calibration.
It is deliberately light on theory, to focus on the usage aspect of the calibration procedures.
For more background on the theory, please refer to  the :doc:`theory section on force calibration</theory/force_calibration/force_calibration>`, while their use is described below.

We can download the data needed for this tutorial directly from Zenodo using Pylake::

    filenames = lk.download_from_doi("10.5281/zenodo.7729823", "test_data")

When force calibration is applied in Bluelake, it uses Pylake to calibrate the force, after which a force calibration item is added to the timeline.
To see what such items looks like, let's load the dataset::

    f = lk.File("test_data/passive_calibration.h5")

The force timeline in this file contains a single calibration measurement.
Note that every force axis (e.g. `1x`, `1y`, `2x` etc) has its own calibration.
We can see calibrations relevant for a single force channel (in this case `1x`) by inspecting the `calibration` attribute for the entire force :class:`~lumicks.pylake.channel.Slice`::

    f.force1x.calibration

This produces a list with the following calibration items:

.. image:: figures/listing.png

This list provides a quick overview which calibration items are present in the file and when they were applied.
More importantly however, it tells us whether the raw data for the calibration is present in the :class:`~lumicks.pylake.channel.Slice`.
When Bluelake creates a calibration item, it only contains the results of a calibration as well as information on when the data was acquired, but *not* the raw data.
We can see this if we plot the time ranges over which these calibration items were acquired and then applied::

    plt.figure(figsize=(8, 2))
    f.force1x.plot(color='k')
    f.force1x.highlight_time_range(f.force1x.calibration[0], color="C0", annotation="0")
    f.force1x.highlight_time_range(f.force1x.calibration[0].applied_at, color="C0")
    f.force1x.highlight_time_range(f.force1x.calibration[1], color="C1", annotation="1")
    f.force1x.highlight_time_range(f.force1x.calibration[1].applied_at, color="C1")

.. image:: figures/time_ranges.png

This helps us see how the calibration items relate to the data present in the file.
Calibration item `0` is the calibration that was acquired *before* the start of this file (and is therefore the calibration that is active when the file starts).
Calibration item `1` is the calibration item acquired in *this* file.

We can see what is in the calibration item by extracting it::

    f.force1x.calibration[1]

.. image:: figures/calibration_item.png

This shows us all the relevant calibration parameters.
These parameters are properties and can be extracted as such::

    >>> calibration_item = f.force1x.calibration[1]
    ... calibration_item.stiffness
    0.1287225353482303

Redoing a Bluelake calibration
------------------------------

We can directly slice the channel by the calibration item we want to reproduce to extract the relevant data::

    force1x_slice = f.force1x[f.force1x.calibration[1]]

To recalibrate data we first have to de-calibrate the data to get back to voltages.
To do this, we divide our data by the force sensitivity that was active at the start of the slice.

    >>> old_calibration = force1x_slice.calibration[0]
    ... volts1x_slice = force1x_slice / old_calibration.force_sensitivity

To reproduce a calibration performed with Bluelake, the easiest way to extract all the relevant parameters is to use :meth:`~lumicks.pylake.force_calibration.calibration_item.ForceCalibrationItem.calibration_params()`::

    >>> calibration_params = f.force1x.calibration[1].calibration_params()
    ... calibration_params
    {'num_points_per_block': 2000,
     'sample_rate': 78125,
     'excluded_ranges': [(19348.0, 19668.0), (24308.0, 24548.0)],
     'fit_range': (100.0, 23000.0),
     'bead_diameter': 4.89,
     'viscosity': 0.00089,
     'temperature': 25.0,
     'fast_sensor': False,
     'axial': False,
     'hydrodynamically_correct': False,
     'active_calibration': False}

This returns a dictionary with exactly those parameters you would need to reproduce this calibration in a format that `pylake` will accept.
Depending on the type of calibration that was performed, the number of parameters may vary.
To quickly perform the same calibration as was performed in Bluelake, we can use the function :func:`~lumicks.pylake.calibrate_force()`.
The easiest way to use this function is to just unpack the dictionary with parameters into it using the `**` notation::

    >>> recalibrated = lk.calibrate_force(volts1x_slice.data, **calibration_params)

We can plot this calibration::

    recalibrated.plot()

.. image:: figures/passive_calibration.png

And the fitting residual:

.. image:: figures/residual.png

We can see that this reproduces the calibration::

    >>> recalibrated.stiffness
    0.12872253516809967

    >>> f.force1x.calibration[1].stiffness
    0.1287225353482303

In this particular case, it looks like we calibrated with the `hydrodynamically_correct` model disabled.
Given that we used big beads, we should have probably enabled it instead.
We can still retroactively change this.

    >>> calibration_params["hydrodynamically_correct"] = True
    ... recalibrated_hyco = lk.calibrate_force(volts1x_slice.data, **calibration_params)
    ... recalibrated_hyco.stiffness
    0.15453110071085924

As expected, the difference in this case is substantial.
