Kymographs
==========

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

To load an HDF5 file and lists all of the kymographs inside of it, run::

    import lumicks.pylake as lk

    file = lk.File("example.h5")
    list(file.kymos)  # e.g. shows: "['reference', 'sytox']"

Once again, `.kymos` is a regular Python dictionary so we can easily iterate over it::

    # Plot all kymos in a file
    for name, kymo in file.kymos.items():
        kymo.plot(channel="rgb")
        plt.savefig(name)

Or just pick a single one::

    kymo = file.kymos["name"]
    kymo.plot("red")

Kymo data and details
---------------------

Access the raw image data::

    rgb = kymo.rgb_image  # matrix with `shape == (h, w, 3)`
    blue = kymo.blue_image  # single color so `shape == (h, w)`

    # Plot manually
    plt.imshow(rgb)

It is possible to crop a kymograph to a specific coordinate range, by using the function :meth:`~lumicks.pylake.Kymo.crop_by_distance`::
For example, we can crop the region from `2` micron to `7` micron using the following command::

    kymo.crop_by_distance(2, 7)

Kymographs can also be sliced in order to obtain a specific time range.
For example, one can plot the region of the kymograph between 175 and 180 seconds using::

    kymo["175s":"180s"].plot("red")

There are also several properties available for convenient access to the kymograph metadata:

* `kymo.center_point_um` provides a dictionary of the central x, y, and z coordinates of the scan in micrometers relative to the brightfield field of view
* `kymo.size_um` provides a list of scan sizes in micrometers along the axes of the scan
* `kymo.pixelsize_um` provides the pixel size in micrometers
* `kymo.pixels_per_line` provides the number of pixels in each line of the kymograph
* `kymo.fast_axis` provides the axis that was scanned (x or y)
* `kymo.line_time_seconds` provides the time between successive lines

By default, kymographs are constructed with units of microns for the position axis. If, however, the kymograph spans a known length of DNA (for example,
lambda DNA) we can calibrate the position axis to kilobase pairs::

    kymo.calibrate_to_kbp(14.850)

Now if we plot the image, the y-axis will be labeled in kbp. These units are also carried forward to any downstream operations such as cropping,
kymotracking algorithms, MSD analysis, etc. *Note: currently this is a static calibration, meaning it is only valid if the traps do not
change position during the time of the kymograph.*

Downsampling kymograph
----------------------

We can downsample a kymograph in time by invoking::

    kymo_ds = kymo.downsampled_by(time_factor=2)

Or in space by invoking::

    kymo_ds = kymo.downsampled_by(position_factor=2)

Or both::

    kymo_ds = kymo.downsampled_by(time_factor=2, position_factor=2)

Note however, that not all functionalities are present anymore when downsampling a kymograph. For
example, if we downsample a kymograph by time, we can no longer access the per pixel timestamps::

    >>> kymo_ds.timestamps
    AttributeError: Per pixel timestamps are no longer available after downsampling a kymograph in time since they are not well defined (the downsampling occurs over a non contiguous time window).
    Line timestamps are still available however. See: `Kymo.line_time_seconds`.

Plotting and exporting
----------------------

There are also convenience functions to plot individual color channels and the full RGB image::

    plt.subplot(2, 1, 1)
    kymo.plot("rgb")
    plt.subplot(2, 1, 2)
    kymo.plot("blue")

The images can also be exported in the TIFF format::

    kymo.save_tiff("image.tiff")
