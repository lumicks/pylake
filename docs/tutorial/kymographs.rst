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
        kymo.plot_rgb()
        plt.savefig(name)

Or just pick a single one::

    kymo = file.kymos["name"]
    kymo.plot_red()

Access the raw image data::

    rgb = kymo.rgb_image  # matrix with `shape == (h, w, 3)`
    blue = kymo.blue_image  # single color so `shape == (h, w)`

    # Plot manually
    plt.imshow(rgb)

The images can also be exported in the TIFF format::

    kymo.save_tiff("image.tiff")

Kymographs can also be sliced in order to obtain a specific time range.
For example, one can plot the region of the kymograph between 175 and 180 seconds using::

    kymo["175s":"180s"].plot_red()

We can obtain the pixel timestamps by invoking `.timestamps`::

    kymo.timestamps

These timestamps are defined as the mean of the timestamps contributing to that pixel.

Kymographs can also be downsampled in case the signal to noise isn't high enough::

    kymo.downsampled_by(2)

Note that for downsampled kymographs, the timestamps are defined as the average of the timestamps corresponding to the
pixels which were summed.
