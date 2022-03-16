Kymographs
==========

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

To load an HDF5 file and lists all of the kymographs inside of it, run::

    import lumicks.pylake as lk

    file = lk.File("example.h5")
    list(file.kymos)  # e.g. shows: "['cas9', 'reference']"

Once again, `.kymos` is a regular Python dictionary so we can easily iterate over it::

    # Plot all kymos in a file
    >>> for name, kymo in file.kymos.items():
            print(f"{name}, starts at {kymo.start} ns")
    cas9, starts at 1586776312560250200 ns
    reference, starts at 1586777007674285402 ns

Or just pick a single one::

    kymo = file.kymos["cas9"]
    kymo.plot(channel="green", aspect="auto", adjustment=lk.ColorAdjustment(0, 5))

.. image:: figures/kymographs/kymo_intro.png

Here we see the `plot()` convenience function. The `channel` argument accepts the strings "red", "green",
"blue", or "rgb". This function accepts keyword arguments that are passed to `plt.imshow()` internally.
Note also, the axes are labeled with the appropriate time and position units.

The kymograph can also be exported to TIFF format::

    kymo.save_tiff("image.tiff")

Kymo data and details
---------------------

We can access the raw image data as `numpy` arrays::

    rgb = kymo.get_image("rgb")  # matrix with `shape == (h, w, 3)`
    blue = kymo.get_image("blue")  # single color so `shape == (h, w)`

    # Plot manually
    plt.imshow(kymo.get_image("green"), aspect="auto", adjustment=lk.ColorAdjustment(0, 5))

.. image:: figures/kymographs/kymo_manual_plotting.png

There are also several properties available for convenient access to the kymograph metadata:

* `kymo.center_point_um` provides a dictionary of the central x, y, and z coordinates of the scan in micrometers relative to the brightfield field of view
* `kymo.size_um` provides a list of scan sizes in micrometers along the axes of the scan
* `kymo.pixelsize_um` provides the pixel size in micrometers
* `kymo.pixels_per_line` provides the number of pixels in each line of the kymograph
* `kymo.fast_axis` provides the axis that was scanned (x or y)
* `kymo.line_time_seconds` provides the time between successive lines

Cropping and slicing
--------------------

It is possible to crop a kymograph to a specific coordinate range, by using the function `Kymo.crop_by_distance`
For example, we can crop the region from `6` micron to `24` micron using the following command::

    kymo.crop_by_distance(6, 24).plot("green)

.. image:: figures/kymographs/kymo_cropped.png

Kymographs can also be sliced in order to obtain a specific time range.
For example, one can plot the region of the kymograph between 114.2 and 164.6 seconds using::

    kymo["114.2s":"164.6s"].plot("green")

.. image:: figures/kymographs/kymo_sliced.png

Note, slicing in time is currently only supported for unprocessed kymographs. If you want to both crop and slice a kymo,
the order of operations is important::

    kymo_sliced = kymo["114.2s":"164.6s"]
    kymo_cropped = kymo_sliced.crop_by_distance(6, 24)

    kymo_cropped.plot("green")

.. image:: figures/kymographs/kymo_cropped_and_sliced.png

Calibrating to base pairs
-------------------------

By default, kymographs are constructed with units of microns for the position axis. If, however, the kymograph spans a known length of DNA (for example,
lambda DNA) we can calibrate the position axis to kilobase pairs::

    kymo_kbp = kymo_cropped.calibrate_to_kbp(48.502)

Now if we plot the image, the y-axis will be labeled in kbp::

    kymo_kbp.plot("green")

.. image:: figures/kymographs/kymo_calibrated.png

These units are also carried forward to any downstream operations such as
kymotracking algorithms and MSD analysis, . *Note: currently this is a static calibration, meaning it is only valid
if the traps do not change position during the time of the kymograph.*

We can also interactively slice, crop, and calibrate kymographs using::

    widget = kymo.crop_and_calibrate(channel="green", tether_length_kbp=48.502)
    plt.show()

.. image:: figures/kymographs/kymo_interactive.png

Simply click and drag the rectangle selector to the desired ROI. After closing the widget, we can access the edited kymograph
with::

    new_kymo = widget.kymo
    new_kymo.plot("green")

.. image:: figures/kymographs/kymo_interactive_result.png

If the optional `tether_length_kbp` argument is supplied, the kymograph is automatically calibrated to the desired
length in kilobase pairs. If this argument is missing (the default value `None`) the edited kymograph is only
sliced and cropped.


Downsampling
------------

We can downsample a kymograph in time by invoking::

    kymo_ds = kymo_cropped.downsampled_by(time_factor=2)

.. image:: figures/kymographs/kymo_downsampled_time.png

Or in space by invoking::

    kymo_ds = kymo_cropped.downsampled_by(position_factor=2)

.. image:: figures/kymographs/kymo_downsampled_position.png

Or both::

    kymo_ds = kymo_cropped.downsampled_by(time_factor=2, position_factor=2)

.. image:: figures/kymographs/kymo_downsampled_time_and_position.png

Note however, that not all functionalities are present anymore when downsampling a kymograph. For
example, if we downsample a kymograph by time, we can no longer access the per pixel timestamps::

    >>> kymo_ds.timestamps
    AttributeError: Per pixel timestamps are no longer available after downsampling a kymograph in time since they
    are not well defined (the downsampling occurs over a non contiguous time window). Line timestamps are still
    available however. See: `Kymo.line_time_seconds`.

Plotting and exporting
----------------------

There are also convenience functions to plot individual color channels and the full RGB image::

    plt.subplot(2, 1, 1)
    kymo.plot("rgb")
    plt.subplot(2, 1, 2)
    kymo.plot("blue")

The images can also be exported in the TIFF format::

    kymo.save_tiff("image.tiff")

Correlating with force
----------------------

We can plot a kymograph along its force trace using::

    kymo.plot_with_force("1x", "green")

This will average the forces over each Kymograph line and plot them in a correlated fashion.
The function can also take a dictionary of extra arguments to customize the kymograph plot.
These parameter values get forwarded to :func:`matplotlib.pyplot.imshow`.
For instance, if a few pixels dominate the image, it might be preferable to set the scale by hand.
This can be accomplished by providing a :class:`~lumicks.pylake.ColorAdjustment`::

    kymo.plot_with_force("1x", "green", adjustment=lk.ColorAdjustment(0, 3))

.. image:: ./figures/kymographs/kymo_correlated.png
